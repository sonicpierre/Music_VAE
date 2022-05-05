import kerastuner as kt
import tensorflow as tf
import wandb
from wandb.keras import WandbCallback

class MyTuner(kt.Tuner):
    """
    Allow you to tune hyperparameters
    """

    def run_trial(self, trial, trainX, batch_size, epochs, objective):

        hp = trial.hyperparameters
        objective_name_str = objective

        ## create the model with the current trial hyperparameters
        model = self.hypermodel.build(hp)
        
        ## Initiates new run for each trial on the dashboard of Weights & Biases
        run = wandb.init(project="Tune Birds Model", config=hp.values)

        ## WandbCallback() logs all the metric data such as
        ## loss, accuracy and etc on dashboard for visualization
        history = model.fit(trainX,
                  trainX,
                  batch_size=batch_size,
                  epochs=epochs,
                  callbacks=[WandbCallback(), tf.keras.callbacks.EarlyStopping('loss', patience=5)])  

        ## if val_accurcy used, use the val_accuracy of last epoch model which is fully trained
        loss = history.history['loss'][-1]  ## [-1] will give the last value in the list

        ## Send the objective data to the oracle for comparison of hyperparameters
        self.oracle.update_trial(trial.trial_id, {objective_name_str:loss})

        ## save the trial model
        self.save_model(trial.trial_id, model)
        
        ## ends the run on the Weights & Biases dashboard
        run.finish()