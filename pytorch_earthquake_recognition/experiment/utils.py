class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __delattr__ = dict.__delitem__

    def __setitem__(self, key, value):
        if isinstance(value, dict):
            value = dotdict(value)
        return dict.__setitem__(self, key, value)


class Evaluator:

    def __init__(self, class_accuracy_dict=None):
        self.class_info = dotdict()  # Javasript style item access

        if class_accuracy_dict:
            for key, value in class_accuracy_dict.items():
                self.class_info[key] = value

    def update_accuracy(self, class_name: str, amount_correct: int, amount_total: int):
        self.class_info[class_name] = {
            'amount_correct': amount_correct,
            'amount_total': amount_total
        }

    def class_details(self, class_name):
        return self.class_info[class_name]

    def percent_correct(self, class_name):
        Class = self.class_info[class_name]
        return Class.amount_correct / Class.amount_total

    def total_percent_correct(self):

        amount_correct = 0
        amount_total = 0

        for class_name, info in self.class_info.items():
            amount_correct += info.amount_correct
            amount_total += info.amount_total

        return amount_correct / amount_total

    def __str__(self):
        return 'Evaluator Object: ' + str(self.class_info)
