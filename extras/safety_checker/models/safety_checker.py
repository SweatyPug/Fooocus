import torch

class SafetyChecker:
    def __init__(self, concept_embeds_weights, special_care_embeds_weights):
        self.concept_embeds_weights = concept_embeds_weights
        self.special_care_embeds_weights = special_care_embeds_weights

    def check_safety(self, images, cos_dist, special_cos_dist, adjustment):
        special_scores = special_cos_dist - self.special_care_embeds_weights + adjustment
        special_care = torch.any(special_scores > 0, dim=1)
        special_adjustment = special_care * 0.01
        special_adjustment = special_adjustment.unsqueeze(1).expand(-1, cos_dist.shape[1])

        concept_scores = (cos_dist - self.concept_embeds_weights) + special_adjustment

        return images

# Example usage:
# safety_checker = SafetyChecker(concept_embeds_weights, special_care_embeds_weights)
# images, has_nsfw_concepts = safety_checker.check_safety(images, cos_dist, special_cos_dist, adjustment)
