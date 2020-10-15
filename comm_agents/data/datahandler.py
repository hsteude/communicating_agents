class DataSet():
    def __init__(self, observations=None, questions=None, opt_answers=None, hidden_states=None):
        self.observations = observations
        self.questions = questions
        self.opt_answers = opt_answers
        self.hidden_states = hidden_states
