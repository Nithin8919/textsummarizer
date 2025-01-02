import torch
import torch.nn.functional as F

def apply_length_penalty(score, length, alpha=1.0):
    return score / ((length + 1) ** alpha)

def beam_search_with_length_penalty(model, input_ids, beam_width=3, max_length=50, alpha=1.0):
    beams = [(input_ids, 0, 0)]
    for _ in range(max_length):
        new_beams = []
        for seq, score, length in beams:
            outputs = model(seq)
            logits = outputs.logits[:, -1, :]
            probs = F.log_softmax(logits, dim=-1)

            top_k_probs, top_k_ids = torch.topk(probs, beam_width)

            for i in range(beam_width):
                new_seq = torch.cat([seq, top_k_ids[:, i].unsqueeze(0)], dim=1)
                new_score = apply_length_penalty(score + top_k_probs[:, i].item(), length + 1, alpha)
                new_beams.append((new_seq, new_score, length + 1))

        beams = sorted(new_beams, key=lambda x: x[1], reverse=True)[:beam_width]
        if all(seq[0, -1] == model.config.eos_token_id for seq, _, _ in beams):
            break

    return beams[0][0]
