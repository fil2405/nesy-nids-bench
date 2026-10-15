from nesybench.metrics import compute_metrics

class EarlyStopper:
    """Early stopping on a trailing mean of the validation score.

    The validation set has few rare-class examples, so the raw macro-F1 moves
    by about 0.01 between epochs. Selecting on the mean of the last `window`
    epochs keeps a lucky epoch from being picked. window=1 is the raw score.
    """

    def __init__(self, patience, window=1):
        if window < 1:
            raise ValueError(f"window must be at least 1, got {window}")
        self.patience = patience
        self.window = window
        self.best_score = -1
        self.best_raw_score = -1
        self.epoch = 0
        self.best_weights = []
        self.counter = 0
        self.best_epoch = 0
        self.scores = []

    def update(self, score, weights):
        self.epoch += 1
        self.scores.append(score)
        recent = self.scores[-self.window:]
        smoothed = sum(recent) / len(recent)

        if smoothed > self.best_score:
            self.best_score = smoothed
            self.best_raw_score = score
            self.best_weights = weights
            self.counter = 0
            self.best_epoch = self.epoch
        else:
            self.counter += 1

        return(self.counter >= self.patience)



def validation_macro_f1(model, splits):

    y_pred = model.predict(splits.X_val, verbose=0, batch_size=1024)

    y_pred = y_pred.argmax(axis=1)

    results = compute_metrics(splits.y_val, y_pred, splits.class_names)

    return results['macro_f1']
