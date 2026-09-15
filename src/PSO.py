# src/PSO.py
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
import pandas as pd
import warnings
warnings.filterwarnings('ignore')


class PSOFeatureSelection:
    def __init__(self,
                 swarm_size=30,
                 iterations=50,
                 w=0.7,
                 c1=1.5,
                 c2=1.5,
                 cv_folds=5,
                 scoring='r2',
                 max_features=None,
                 random_seed=42,
                 threshold=0.5):
        self.swarm_size = swarm_size
        self.iterations = iterations
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.cv_folds = cv_folds
        self.scoring = scoring
        self.max_features = max_features
        self.random_seed = random_seed
        self.threshold = threshold
        np.random.seed(random_seed)

        self.X = None
        self.y = None
        self.feature_names = None
        self.n_features = None

        self.particle_pos = None
        self.particle_vel = None
        self.particle_best_pos = None
        self.particle_best_score = None
        self.global_best_pos = None
        self.global_best_score = -np.inf
        self.history = []

    def _decode_position(self, pos):
        if self.max_features is not None:
            k = min(self.max_features, len(pos))
            mask = np.zeros(len(pos), dtype=int)
            top_k_idx = np.argsort(pos)[-k:]
            mask[top_k_idx] = 1
            return mask
        else:
            return (pos >= self.threshold).astype(int)

    def _count_selected(self, mask):
        return np.sum(mask)

    def _evaluate_fitness(self, pos):
        mask = self._decode_position(pos)
        n_selected = self._count_selected(mask)
        if n_selected == 0:
            return -1e9
        X_subset = self.X[:, mask.astype(bool)]
        model = RandomForestRegressor(
            n_estimators=30, max_depth=10,
            random_state=self.random_seed, n_jobs=-1
        )
        scores = cross_val_score(model, X_subset, self.y,
                                 cv=self.cv_folds, scoring=self.scoring)
        return np.mean(scores)

    def _initialize_swarm(self):
        n = self.n_features
        self.particle_pos = np.random.rand(self.swarm_size, n)
        self.particle_vel = np.random.uniform(-0.1, 0.1, (self.swarm_size, n))
        self.particle_best_score = np.array(
            [self._evaluate_fitness(p) for p in self.particle_pos]
        )
        self.particle_best_pos = self.particle_pos.copy()
        best_idx = np.argmax(self.particle_best_score)
        self.global_best_pos = self.particle_pos[best_idx].copy()
        self.global_best_score = self.particle_best_score[best_idx]

    def optimize(self, X, y, feature_names=None):
        if isinstance(X, pd.DataFrame):
            self.feature_names = X.columns.tolist()
            X = X.values
        else:
            self.feature_names = ([f'Feature_{i}' for i in range(X.shape[1])]
                                  if feature_names is None else feature_names)

        self.X = np.array(X, dtype=float)
        self.y = np.array(y)
        self.n_features = self.X.shape[1]

        print(f"[PSO] Descriptores: {self.n_features}, "
              f"Partículas: {self.swarm_size}, Iteraciones: {self.iterations}")

        self._initialize_swarm()

        for it in range(self.iterations):
            for i in range(self.swarm_size):
                current_score = self._evaluate_fitness(self.particle_pos[i])
                if current_score > self.particle_best_score[i]:
                    self.particle_best_score[i] = current_score
                    self.particle_best_pos[i] = self.particle_pos[i].copy()
                if current_score > self.global_best_score:
                    self.global_best_score = current_score
                    self.global_best_pos = self.particle_pos[i].copy()

            for i in range(self.swarm_size):
                r1 = np.random.rand(self.n_features)
                r2 = np.random.rand(self.n_features)
                cognitive = self.c1 * r1 * (
                    self.particle_best_pos[i] - self.particle_pos[i]
                )
                social = self.c2 * r2 * (
                    self.global_best_pos - self.particle_pos[i]
                )
                self.particle_vel[i] = (
                    self.w * self.particle_vel[i] + cognitive + social
                )
                self.particle_pos[i] = self.particle_pos[i] + self.particle_vel[i]
                self.particle_pos[i] = np.clip(self.particle_pos[i], 0, 1)

            best_mask = self._decode_position(self.global_best_pos)
            self.history.append({
                'iteration': it + 1,
                'best_score': self.global_best_score,
                'n_selected': int(np.sum(best_mask)),
            })

            if (it + 1) % 10 == 0 or it == 0:
                print(f"[PSO] Iter {it+1}/{self.iterations} "
                      f"- Mejor R²: {self.global_best_score:.4f} "
                      f"- Descriptores: {int(np.sum(best_mask))}")

        best_mask = self._decode_position(self.global_best_pos)
        best_features = [self.feature_names[i]
                         for i in range(self.n_features) if best_mask[i] == 1]

        print(f"[PSO] ¡Listo! R²={self.global_best_score:.4f}, "
              f"seleccionados={len(best_features)}")
        return best_mask, self.global_best_score, best_features, self.history