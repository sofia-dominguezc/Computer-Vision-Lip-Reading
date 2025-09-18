# Enhancing Visual Speech Recognition with Language Models
By Sofia Dominguez (domingc7@mit.edu) and Daria Kryvosheieva (daria_k@mit.edu)

This project fine-tuned <a href="https://ieeexplore.ieee.org/document/10096889">Auto-AVSR</a> on two additional datasets (<a href="https://ieeexplore.ieee.org/document/7050271">TCD-TIMIT</a> + <a href="https://ieeexplore.ieee.org/document/10483898">WildVSR</a>) and experimented with integrating a language model (<a href="https://huggingface.co/google/gemma-3-12b-it">Gemma-3-12b</a>) into the beam search decoding process.

## Abstract

Lip reading, or visual speech recognition (VSR), is essential for enhancing communication in hearing-impaired individuals and improving speech comprehension in noisy environments. However, current VSR systems struggle with the inherent ambiguity of visually similar lip movements and the scarcity of large, high-quality training datasets. To address these issues, we propose enhancing the state-of-the art model, Auto-AVSR, with two key modifications: (1) integrating an LLM into the decoding process to decrease ambiguity by utilizing context, and (2) fine-tuning on previously unused datasets to increase training diversity. With these changes, our approach aims to advance lip reading performance, which allows for more reliable VSR applications.

Read full report [here](6_S058_Project_Report.pdf).
