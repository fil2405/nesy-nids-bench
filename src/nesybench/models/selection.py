from nesybench.metrics import compute_metrics

class EarlyStopper:

    def __init__(self, patience):
        self.patience = patience
        self.best_score = -1
        self.epoch = 0
        self.best_weights = []
        self.counter = 0
        self.best_epoch = 0

    def update(self, score, weights):
        self.epoch += 1

        if score > self.best_score:
            self.best_score = score
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
