# 5-minute demo script

Fill the `[ ]` blanks with YOUR numbers from `results/summary.md` after training. Never say a number you did not measure.

## Before you walk in
- Open the deployed app 30 min early (free tier sleeps). Also have `streamlit run app.py` ready on your laptop.
- Have the recorded demo video and `results/curves.png` open as a fallback.
- Pick 2 test images and 2 settings you already know look good.

## Script

**0:00 Problem (30 s)**
"When an image travels over a telecom link it gets damaged: noise, lost packets, compression. I built a system that
simulates that channel and compares two ways to repair the image: classic OpenCV filters and a small neural network."

**0:30 Method (45 s)** (one slide: image → channel → restoration → PSNR/SSIM)
"The channel has five damage types, each with a severity slider. The classic baseline is told what damage happened.
My network is blind: one model, trained on mixed damage, with no hint. I measure PSNR and SSIM against the original."

**1:15 Live demo (2 min)** in the app
1. Choose a sample image. Damage type: **Packet loss**, severity 0.6. "These black blocks are packets that never arrived."
2. Point to the panels: Damaged [ ] dB → Classic [ ] dB → AI [ ] dB.
3. Switch to **Gaussian noise**, push severity to the maximum. "At [ ] dB SNR the image is barely recognisable."
   Compare classic vs AI. Say honestly which one wins here.
4. Switch to **Combined**, the realistic case: compression, noise and loss together.
5. Upload an image from the audience / the teacher's laptop to show it is not cherry-picked.

**3:15 Results (1 min)** `Quality curve` tab or `results/curves.png`
"Quality versus channel severity. The AI beats classic on [ ], and classic wins on [ ] because [ ]."

**4:15 Pipeline and limits (45 s)**
"Trained on Colab, exported to ONNX, committed to GitHub, auto-deployed on Streamlit. Limits: it only knows the damage
types it trained on, and the baseline gets the lost-block map for free."

**5:00 End.** Stop talking and invite questions.

## Questions you are likely to get

**Why PSNR and SSIM?** PSNR measures pixel error in dB (standard in telecom). SSIM measures structural similarity and
tracks human perception better. I report both.

**Why does classic win (or the AI lose) on X?** Classic is told the damage type and for packet loss gets the exact lost
block positions. The CNN is blind. A fair, honest comparison, not a weak baseline.

**What is SNR?** Signal power divided by noise power, in dB. 30 dB = light noise, 5 dB = very noisy.

**Why a residual network?** It predicts the correction (damage) and adds it back to the input. That is easier to learn
and trains faster than predicting the whole image.

**Why dilated convolutions?** They widen the receptive field (~49 px) cheaply, so the network can see across a lost 8×8
block to fill it in.

**Could this work on real transmission data?** Not yet. The channel is simulated. Real channels have burst errors and
other patterns; training on those is future work.

**Why ONNX?** Small, fast on CPU and no PyTorch needed in the deployed app, so the free hosting tier handles it.

**What would you improve?** More damage types, a U-Net, larger training set, a blind detector for lost blocks so the
CNN needs no side information, evaluation on a bigger test set.
