from abc import ABC, abstractmethod

class BaseTokenizer(ABC):
    def __init__(self, subject_id, model_weights='', redo_tokenization=False):
        self.subject_id = subject_id
        self.model_weights = model_weights
        self.redo_tokenization = redo_tokenization
        self.tokenization_dict = {'input': None, 'target': None}
        self.input_window_size = None
        self.target_window_size = None
        self.input_channel_names = None
        self.target_channel_names = None

    @abstractmethod
    def fit(self, train_input_signal, train_target_signal, input_token_size, target_token_size,
            input_sample_rate, target_sample_rate, input_channel_names, target_channel_names,
            input_t_min=0, input_t_max=1, target_t_min=0, target_t_max=1):
        pass

    @abstractmethod
    def tokenize(self, signal_type, data):
        pass

    @abstractmethod
    def inverse_tokenize(self, signal_type, data):
        pass

    @abstractmethod
    def plot_components(self, signal_type, path=''):
        pass

    @abstractmethod
    def plot_explained_variance(self, signal_type, path=''):
        pass