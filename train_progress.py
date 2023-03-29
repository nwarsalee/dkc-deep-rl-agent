from stable_baselines3.common.callbacks import BaseCallback
import os

class TrainingCallback(BaseCallback):
    # Callback class used for the model on finishing a step, determines what to occur on a step finish
    def __init__(self, frequency, dir_path, verbose=1):
        super(TrainingCallback, self).__init__(verbose)
        # frequency - the amount of steps to be performed before saving the model to a file
        # dir_path - the location to store the model file 
        self.frequency = frequency
        self.dir_path = dir_path

    def _init_callback(self):
        # create the dir_path directory if it does not exist
        if self.dir_path is not None:
            os.makedirs(self.dir_path, exist_ok=True)

    def _on_step(self):
        # On a step, determine if the frequency has been reached and if so, save the model
        if self.n_calls % self.frequency == 0:
            model_path = os.path.join(self.dir_path, 'DKModel_{}'.format(self.n_calls))
            self.model.save(model_path)
        return True