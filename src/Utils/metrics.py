import torch

class PrivacyLeakageMetric:
    def __init__(self):
        self.total_leakage = 0.0
        self.num_samples = 0

    def update(self, pred, target):
        """
        Update the metric with new predictions and targets.
        :param pred: Model predictions (e.g., inferred user preferences).
        :param target: Ground truth (e.g., actual user preferences).
        """
        leakage = torch.abs(pred - target).sum().item()
        self.total_leakage += leakage
        self.num_samples += 1

    def compute(self):
        """
        Compute the average privacy leakage.
        :return: Average leakage per sample.
        """
        if self.num_samples == 0:
            return 0.0
        return self.total_leakage / self.num_samples


class MembershipInferenceAttackSuccessRate:
    def __init__(self):
        self.correct = 0
        self.total = 0

    def update(self, pred, target):
        """
        Update the metric with new predictions and targets.
        :param pred: Model predictions (e.g., confidence scores for membership).
        :param target: Ground truth (1 if the sample is in the training set, 0 otherwise).
        """
        predicted_membership = (pred > 0.5).int()
        self.correct += (predicted_membership == target).sum().item()
        self.total += target.size(0)

    def compute(self):
        """
        Compute the success rate of the membership inference attack.
        :return: Success rate as a percentage.
        """
        if self.total == 0:
            return 0.0
        return (self.correct / self.total) * 100.0


class RobustAccuracy:
    def __init__(self):
        self.correct = 0
        self.total = 0

    def update(self, pred, target):
        """
        Update the metric with new predictions and targets.
        :param pred: Model predictions (e.g., class labels or scores).
        :param target: Ground truth labels.
        """
        predicted_labels = pred.argmax(dim=1)
        self.correct += (predicted_labels == target).sum().item()
        self.total += target.size(0)

    def compute(self):
        """
        Compute the robust accuracy.
        :return: Accuracy as a percentage.
        """
        if self.total == 0:
            return 0.0
        return (self.correct / self.total) * 100.0


class AttackSuccessRate:
    def __init__(self):
        self.successful_attacks = 0
        self.total_attacks = 0

    def update(self, pred, target):
        """
        Update the metric with new predictions and targets.
        :param pred: Model predictions on adversarial examples.
        :param target: Ground truth labels.
        """
        predicted_labels = pred.argmax(dim=1)
        self.successful_attacks += (predicted_labels != target).sum().item()
        self.total_attacks += target.size(0)

    def compute(self):
        """
        Compute the attack success rate.
        :return: Success rate as a percentage.
        """
        if self.total_attacks == 0:
            return 0.0
        return (self.successful_attacks / self.total_attacks) * 100.0


class PerturbationSize:
    def __init__(self):
        self.total_perturbation = 0.0
        self.num_samples = 0

    def update(self, pred, target):
        """
        Update the metric with new predictions and targets.
        :param pred: Model predictions on adversarial examples.
        :param target: Ground truth labels.
        """
        perturbation = torch.norm(pred - target, p=2).item()
        self.total_perturbation += perturbation
        self.num_samples += 1

    def compute(self):
        """
        Compute the average perturbation size.
        :return: Average perturbation size.
        """
        if self.num_samples == 0:
            return 0.0
        return self.total_perturbation / self.num_samples