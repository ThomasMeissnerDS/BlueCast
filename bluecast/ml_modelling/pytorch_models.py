import logging
import time

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import copy

logger = logging.getLogger(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class _PyTorchBaseEstimator(BaseEstimator):
    """Base estimator for PyTorch models with budget-aware training.

    """

    def __init__(
        self,
        hidden_layer_sizes=(100,),
        activation="relu",
        alpha=0.0001,
        learning_rate_init=0.001,
        max_iter=200,
        batch_size=256,
        early_stopping_rounds=10,
        random_state=None,
        l1_ratio=0.0,
        dropout_rate=0.0,
        scoring="neg_mean_squared_error",
        training_deadline=None,
        **kwargs,
    ):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation = activation
        self.alpha = alpha
        self.learning_rate_init = learning_rate_init
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.early_stopping_rounds = early_stopping_rounds
        self.random_state = random_state
        self.l1_ratio = l1_ratio
        self.dropout_rate = dropout_rate
        self.scoring = scoring
        self.training_deadline = training_deadline
        
        self.model_ = None
        self.classes_ = None
        self.convergence_info_ = {}

    def _build_model(self, input_dim, output_dim):
        if self.random_state is not None:
            torch.manual_seed(self.random_state)
            
        layers = []
        in_dim = input_dim
            
        for hidden_dim in self.hidden_layer_sizes:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU() if self.activation == "relu" else nn.Tanh())
            if self.dropout_rate > 0:
                layers.append(nn.Dropout(p=self.dropout_rate))
            in_dim = hidden_dim
            
        layers.append(nn.Linear(in_dim, output_dim))
        
        model = nn.Sequential(*layers).to(device)
        
        # Weight initialization: He for ReLU, Xavier for Tanh
        for m in model.modules():
            if isinstance(m, nn.Linear):
                if self.activation == "relu":
                    nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                else:
                    nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)
        
        return model

    def fit(self, X, y):
        # Implementation in subclasses
        pass

    def _train_loop(self, X_train, y_train, is_classifier=False):
        if self.random_state is not None:
            torch.manual_seed(self.random_state)
            np.random.seed(self.random_state)
            
        # Split a small validation set for early stopping (10%)
        val_size = max(32, int(0.15 * len(X_train)))
        indices = np.random.permutation(len(X_train))
        train_idx, val_idx = indices[val_size:], indices[:val_size]
        
        X_t = torch.tensor(X_train[train_idx], dtype=torch.float32)
        X_v = torch.tensor(X_train[val_idx], dtype=torch.float32)
        
        if is_classifier:
            y_t = torch.tensor(y_train[train_idx], dtype=torch.long)
            y_v = torch.tensor(y_train[val_idx], dtype=torch.long)
            criterion = nn.CrossEntropyLoss()
            output_dim = len(self.classes_) if len(self.classes_) > 2 else 2
        else:
            y_t = torch.tensor(y_train[train_idx], dtype=torch.float32).view(-1, 1)
            y_v = torch.tensor(y_train[val_idx], dtype=torch.float32).view(-1, 1)
            
            from bluecast.ai.metrics import get_pytorch_loss_from_scoring
            criterion = get_pytorch_loss_from_scoring(self.scoring)
            
            output_dim = 1
            
        train_dataset = TensorDataset(X_t, y_t)
        actual_batch_size = min(self.batch_size, len(X_t))
        train_loader = DataLoader(train_dataset, batch_size=actual_batch_size, shuffle=True)
        
        self.model_ = self._build_model(X_train.shape[1], output_dim)
        
        # Optimizer with L2 regularization (weight_decay)
        # alpha is total penalty. L2 penalty is alpha * (1 - l1_ratio)
        l2_penalty = self.alpha * (1 - self.l1_ratio)
        optimizer = optim.AdamW(self.model_.parameters(), lr=self.learning_rate_init, weight_decay=l2_penalty)
        
        patience_lr = max(2, self.early_stopping_rounds // 3)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=patience_lr)
        
        best_val_loss = float('inf')
        best_model_wts = copy.deepcopy(self.model_.state_dict())
        patience_counter = 0
        val_losses = []
        budget_exhausted = False
        final_epoch = 0
        
        for epoch in range(self.max_iter):
            if self.training_deadline and time.time() > self.training_deadline:
                budget_exhausted = True
                break
                
            final_epoch = epoch
            self.model_.train()
            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                optimizer.zero_grad()
                outputs = self.model_(batch_x)
                loss = criterion(outputs, batch_y)
                
                # L1 regularization
                if self.l1_ratio > 0:
                    l1_penalty = 0
                    for param in self.model_.parameters():
                        l1_penalty += torch.sum(torch.abs(param))
                    loss += self.alpha * self.l1_ratio * l1_penalty
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model_.parameters(), max_norm=1.0)
                optimizer.step()
                
            self.model_.eval()
            with torch.no_grad():
                val_outputs = self.model_(X_v.to(device))
                val_loss = criterion(val_outputs, y_v.to(device)).item()

            val_losses.append(val_loss)
                
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_wts = copy.deepcopy(self.model_.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1
                
            scheduler.step(val_loss)
                
            if patience_counter >= self.early_stopping_rounds:
                break
                
        self.model_.load_state_dict(best_model_wts)

        # Store convergence diagnostics
        self.convergence_info_ = {
            "epochs_run": final_epoch + 1,
            "max_iter": self.max_iter,
            "converged": patience_counter >= self.early_stopping_rounds,
            "budget_exhausted": budget_exhausted,
            "best_val_loss": float(best_val_loss),
            "last_5_val_losses": [float(v) for v in val_losses[-5:]],
            "final_lr": optimizer.param_groups[0]["lr"],
        }
        return self


class PyTorchMLPRegressor(_PyTorchBaseEstimator, RegressorMixin):
    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)
        return self._train_loop(X, y, is_classifier=False)
        
    def predict(self, X):
        X = np.asarray(X)
        self.model_.eval()
        with torch.no_grad():
            X_t = torch.tensor(X, dtype=torch.float32).to(device)
            preds = self.model_(X_t).cpu().numpy().ravel()
        return preds


class PyTorchMLPClassifier(_PyTorchBaseEstimator, ClassifierMixin):
    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        # Map classes to 0, 1, 2...
        y_mapped = np.searchsorted(self.classes_, y)
        return self._train_loop(X, y_mapped, is_classifier=True)
        
    def predict_proba(self, X):
        X = np.asarray(X)
        self.model_.eval()
        with torch.no_grad():
            X_t = torch.tensor(X, dtype=torch.float32).to(device)
            logits = self.model_(X_t)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
        return probs
        
    def predict(self, X):
        probs = self.predict_proba(X)
        return self.classes_[np.argmax(probs, axis=1)]


# ---------------------------------------------------------------------------
# SoftOrdering1DCNN — 1D CNN for tabular data
# ---------------------------------------------------------------------------


class _SoftOrdering1DCNN(nn.Module):
    """1D CNN that learns a soft ordering of input features via a dense
    projection followed by grouped and depthwise convolutions with a
    residual connection.

    Reference: https://arxiv.org/abs/2104.06321

    :param input_dim: Number of input features.
    :param output_dim: Number of outputs (1 for regression, n_classes for classification).
    :param sign_size: Signature size — controls the spatial dimension after the initial projection.
    :param cha_input: Number of input channels for the conv layers.
    :param cha_hidden: Number of hidden channels for deeper conv layers.
    :param K: Channel expansion factor in the first grouped convolution.
    :param dropout_input: Dropout rate applied after the input batch norm.
    :param dropout_hidden: Dropout rate applied in the hidden conv blocks.
    :param dropout_output: Dropout rate applied before the final dense layer.
    """

    def __init__(
        self,
        input_dim,
        output_dim=1,
        sign_size=32,
        cha_input=16,
        cha_hidden=32,
        K=2,
        dropout_input=0.2,
        dropout_hidden=0.2,
        dropout_output=0.2,
    ):
        super().__init__()

        hidden_size = sign_size * cha_input
        sign_size1 = sign_size
        sign_size2 = sign_size // 2
        output_size = (sign_size // 4) * cha_hidden

        self.cha_input = cha_input
        self.sign_size1 = sign_size1

        self.batch_norm1 = nn.BatchNorm1d(input_dim)
        self.dropout1 = nn.Dropout(dropout_input)
        dense1 = nn.Linear(input_dim, hidden_size, bias=False)
        self.dense1 = nn.utils.weight_norm(dense1)

        # 1st conv layer — grouped (depthwise)
        self.batch_norm_c1 = nn.BatchNorm1d(cha_input)
        conv1 = nn.Conv1d(
            cha_input, cha_input * K, kernel_size=5, stride=1, padding=2,
            groups=cha_input, bias=False,
        )
        self.conv1 = nn.utils.weight_norm(conv1, dim=None)
        self.ave_po_c1 = nn.AdaptiveAvgPool1d(output_size=sign_size2)

        # 2nd conv layer
        self.batch_norm_c2 = nn.BatchNorm1d(cha_input * K)
        self.dropout_c2 = nn.Dropout(dropout_hidden)
        conv2 = nn.Conv1d(
            cha_input * K, cha_hidden, kernel_size=3, stride=1, padding=1, bias=False,
        )
        self.conv2 = nn.utils.weight_norm(conv2, dim=None)

        # 3rd conv layer
        self.batch_norm_c3 = nn.BatchNorm1d(cha_hidden)
        self.dropout_c3 = nn.Dropout(dropout_hidden)
        conv3 = nn.Conv1d(
            cha_hidden, cha_hidden, kernel_size=3, stride=1, padding=1, bias=False,
        )
        self.conv3 = nn.utils.weight_norm(conv3, dim=None)

        # 4th conv layer — grouped (depthwise) + residual
        self.batch_norm_c4 = nn.BatchNorm1d(cha_hidden)
        conv4 = nn.Conv1d(
            cha_hidden, cha_hidden, kernel_size=5, stride=1, padding=2,
            groups=cha_hidden, bias=False,
        )
        self.conv4 = nn.utils.weight_norm(conv4, dim=None)
        self.avg_po_c4 = nn.AvgPool1d(kernel_size=4, stride=2, padding=1)

        self.flt = nn.Flatten()

        self.batch_norm2 = nn.BatchNorm1d(output_size)
        self.dropout2 = nn.Dropout(dropout_output)
        dense2 = nn.Linear(output_size, output_dim, bias=False)
        self.dense2 = nn.utils.weight_norm(dense2)

    def forward(self, x):
        x = self.batch_norm1(x)
        x = self.dropout1(x)
        x = nn.functional.celu(self.dense1(x))

        x = x.reshape(x.shape[0], self.cha_input, self.sign_size1)

        x = self.batch_norm_c1(x)
        x = nn.functional.relu(self.conv1(x))
        x = self.ave_po_c1(x)

        x = self.batch_norm_c2(x)
        x = self.dropout_c2(x)
        x = nn.functional.relu(self.conv2(x))
        x_s = x  # residual

        x = self.batch_norm_c3(x)
        x = self.dropout_c3(x)
        x = nn.functional.relu(self.conv3(x))

        x = self.batch_norm_c4(x)
        x = self.conv4(x)
        x = x + x_s  # residual connection
        x = nn.functional.relu(x)

        x = self.avg_po_c4(x)
        x = self.flt(x)

        x = self.batch_norm2(x)
        x = self.dropout2(x)
        x = self.dense2(x)
        return x


class _PyTorchSO1DCNNBase(_PyTorchBaseEstimator):
    """Base estimator that uses SoftOrdering1DCNN instead of an MLP.

    Adds SO1DCNN-specific hyperparameters (sign_size, cha_input, cha_hidden, K)
    while reusing the entire training loop infrastructure from _PyTorchBaseEstimator.
    """

    def __init__(
        self,
        sign_size=32,
        cha_input=16,
        cha_hidden=32,
        K=2,
        dropout_input=0.2,
        dropout_hidden=0.2,
        dropout_output=0.2,
        alpha=0.0001,
        learning_rate_init=0.001,
        max_iter=200,
        batch_size=256,
        early_stopping_rounds=10,
        random_state=None,
        l1_ratio=0.0,
        scoring="neg_mean_squared_error",
        **kwargs,
    ):
        # Pass through to base — hidden_layer_sizes/activation unused but kept for API compat
        super().__init__(
            hidden_layer_sizes=(),
            activation="relu",
            alpha=alpha,
            learning_rate_init=learning_rate_init,
            max_iter=max_iter,
            batch_size=batch_size,
            early_stopping_rounds=early_stopping_rounds,
            random_state=random_state,
            l1_ratio=l1_ratio,
            dropout_rate=0.0,  # SO1DCNN manages its own dropout
            scoring=scoring,
        )
        self.sign_size = sign_size
        self.cha_input = cha_input
        self.cha_hidden = cha_hidden
        self.K = K
        self.dropout_input = dropout_input
        self.dropout_hidden = dropout_hidden
        self.dropout_output = dropout_output

    def _build_model(self, input_dim, output_dim):
        if self.random_state is not None:
            torch.manual_seed(self.random_state)

        model = _SoftOrdering1DCNN(
            input_dim=input_dim,
            output_dim=output_dim,
            sign_size=self.sign_size,
            cha_input=self.cha_input,
            cha_hidden=self.cha_hidden,
            K=self.K,
            dropout_input=self.dropout_input,
            dropout_hidden=self.dropout_hidden,
            dropout_output=self.dropout_output,
        ).to(device)
        return model


class PyTorchSO1DCNNRegressor(_PyTorchSO1DCNNBase, RegressorMixin):
    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)
        return self._train_loop(X, y, is_classifier=False)

    def predict(self, X):
        X = np.asarray(X)
        self.model_.eval()
        with torch.no_grad():
            X_t = torch.tensor(X, dtype=torch.float32).to(device)
            preds = self.model_(X_t).cpu().numpy().ravel()
        return preds


class PyTorchSO1DCNNClassifier(_PyTorchSO1DCNNBase, ClassifierMixin):
    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        y_mapped = np.searchsorted(self.classes_, y)
        return self._train_loop(X, y_mapped, is_classifier=True)

    def predict_proba(self, X):
        X = np.asarray(X)
        self.model_.eval()
        with torch.no_grad():
            X_t = torch.tensor(X, dtype=torch.float32).to(device)
            logits = self.model_(X_t)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
        return probs

    def predict(self, X):
        probs = self.predict_proba(X)
        return self.classes_[np.argmax(probs, axis=1)]

