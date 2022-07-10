import torch
from CMA_MIL import CMAL
import math


def avce_train(dataloader, model_av, model_v, optimizer_av, optimizer_v, criterion, lamda_a2b, lamda_a2n, logger):
    with torch.set_grad_enabled(True):
        model_av.train()
        model_v.train()
        for i, (f_v, f_a, label) in enumerate(dataloader):
            seq_len = torch.sum(torch.max(torch.abs(f_v), dim=2)[0] > 0, 1)
            f_v = f_v[:, :torch.max(seq_len), :]
            f_a = f_a[:, :torch.max(seq_len), :]
            f_v, f_a, label = f_v.float().cuda(), f_a.float().cuda(), label.float().cuda()
            mmil_logits, audio_logits, visual_logits, _, audio_rep, visual_rep = model_av(f_a, f_v, seq_len)
            audio_logits = audio_logits.squeeze()
            visual_logits = visual_logits.squeeze()
            mmil_logits = mmil_logits.squeeze()
            clsloss = criterion(mmil_logits, label)
            cmaloss_a2v_a2b, cmaloss_a2v_a2n, cmaloss_v2a_a2b, cmaloss_v2a_a2n = CMAL(mmil_logits, audio_logits,
                                                                                      visual_logits, seq_len, audio_rep,
                                                                                      visual_rep)
            total_loss = clsloss + lamda_a2b * cmaloss_a2v_a2b + lamda_a2b * cmaloss_v2a_a2b + lamda_a2n * cmaloss_a2v_a2n + lamda_a2n * cmaloss_v2a_a2n
            unit = dataloader.__len__() // 2
            if i % unit == 0:
                logger.info(f"Current Lambda_a2b: {lamda_a2b:.2f}, Current Lambda_a2n: {lamda_a2n:.2f}")
                logger.info(
                    f"{int(i // unit)}/{2} MIL Loss: {clsloss:.4f}, CMA Loss A2V_A2B: {cmaloss_a2v_a2b:.4f}, CMA Loss A2V_A2N: {cmaloss_a2v_a2n:.4f},"
                    f"CMA Loss V2A_A2B: {cmaloss_v2a_a2b:.4f},  CMA Loss V2A_A2N: {cmaloss_v2a_a2n:.4f}")

            v_logits = model_v(f_v, seq_len)
            loss_v = criterion(v_logits, label)

            optimizer_av.zero_grad()
            optimizer_v.zero_grad()
            model_av.requires_grad = True
            model_v.requires_grad = False
            total_loss.backward()
            optimizer_av.step()

            optimizer_av.zero_grad()
            optimizer_v.zero_grad()
            model_av.requires_grad = False
            model_v.requires_grad = True
            loss_v.backward()
            optimizer_v.step()

        return total_loss, loss_v
