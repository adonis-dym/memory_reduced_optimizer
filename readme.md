# Reducing Memory Footprint in Deep Network Training by Gradient Space Reutilization

This repository contains the source code for the paper "Reducing Memory Footprint in Deep Network Training by Gradient Space Reutilization," which proposes to reuse the oldest gradient space for storing the intermediate variables once it is no longer needed. We apply this method on several mainstream optimizers to get the memory reduced variants, named Adam_R, Adan_R, and Lion_R, respectively.

## Experimental Results
The experimental results demonstrate the efficacy of our memory reduction strategies across various model architectures. Below are the summarized results showing the peak memory usage and savings achieved with the memory-reduced variants of the optimizers.

### Memory Usage and Savings for AdamW and AdamW-R
| Model         | # Params | AdamW (MB) | AdamW-R (MB) | Savings (%) | ZeRO       |
|---------------|----------|------------|--------------|-------------|------------|
| ViT-S         | 22.9M    | 526        | 417          | **20.71**   | ✗          |
| ViT-B         | 88.2M    | 2007       | 1629         | **18.81**   | ✗          |
| ViT-L         | 305.5M   | 6367       | 5046         | **20.75**   | ✗          |
| ViT-H         | 630.8M   | 13336      | 10777        | **19.19**   | ✗          |
| ViT-G         | 1.0B     | 21542      | 17408        | **19.19**   | ✗          |
| ConvNeXt-T    | 28.6M    | 684        | 621          | **9.20**    | ✗          |
| ConvNeXt-S    | 50.2M    | 1177       | 1009         | **14.26**   | ✗          |
| ConvNeXt-B    | 88.6M    | 1894       | 1629         | **13.95**   | ✗          |
| ConvNeXt-L    | 197.8M   | 4387       | 3706         | **15.54**   | ✗          |
| ConvNeXt-XL   | 350.2M   | 7218       | 6004         | **16.82**   | ✗          |
| BLOOM-560M    | 559.2M   | 15531      | 13822        | **11.00**   | ✗          |
| BLOOM-560M    | 559.2M   | 5339       | 5011         | **6.15**    | ✓          |
| BLOOM-3B      | 3.0B     | 23477      | 21964        | **6.45**    | ✓          |
| BLOOM-7B      | 7.1B     | 44826      | 41296        | **7.87**    | ✓          |
| Phi-1.5       | 1.4B     | 36650      | 36008        | 1.75        | ✗          |
| Phi-1.5       | 1.4B     | 18616      | 17949        | **3.59**    | ✓          |
| Phi-2         | 2.8B     | 27581      | 26132        | **5.26**    | ✓          |
| Qwen-0.5B     | 464.0M   | 12581      | 11272        | **10.40**   | ✗          |
| Qwen-0.5B     | 464.0M   | 4897       | 4837         | 1.23        | ✓          |
| Qwen-1.8B     | 1.8B     | 46410      | 38986        | **16.00**   | ✗          |
| Qwen-1.8B     | 1.8B     | 12756      | 11902        | **6.69**    | ✓          |
| LLaMA-2-7B    | 6.7B     | 32325      | 29002        | **10.28**   | ✓          |
| LLaMA-2-13B   | 13.0B    | 49103      | 45768        | **6.79**    | ✓          |
| Gemma-2B      | 2.5B     | 19609      | 18365        | **6.35**    | ✓          |
| Gemma-7B      | 8.5B     | 47029      | 42841        | **8.90**    | ✓          |
| Vicuna-7B     | 6.7B     | 32351      | 28993        | **10.38**   | ✓          |
| Vicuna-13B    | 13.0B    | 49327      | 46089        | **6.57**    | ✓          |
| ChatGLM3-6B   | 6.2B     | 31491      | 28369        | **9.92**    | ✓          |
| Falcon-7B     | 6.9B     | 33643      | 30168        | **10.33**   | ✓          |

### Memory Usage and Savings for Adan and Adan-R
| Model         | # Params | Adan (MB) | Adan-R (MB) | Savings (%) | ZeRO       |
|---------------|----------|-----------|-------------|-------------|------------|
| ViT-S         | 22.9M    | 711       | 621         | **12.68**   | ✗          |
| ViT-B         | 88.2M    | 2806      | 2407        | **14.20**   | ✗          |
| ViT-L         | 305.5M   | 8812      | 7491        | **14.99**   | ✗          |
| ViT-H         | 630.8M   | 18639     | 16110       | **13.57**   | ✗          |
| ViT-G         | 1.0B     | 30130     | 25910       | **14.00**   | ✗          |
| ConvNeXt-T    | 28.6M    | 927       | 864         | **6.78**    | ✗          |
| ConvNeXt-S    | 50.2M    | 1634      | 1466        | **10.27**   | ✗          |
| ConvNeXt-B    | 88.6M    | 2632      | 2355        | **10.52**   | ✗          |
| ConvNeXt-L    | 197.8M   | 6078      | 5417        | **10.87**   | ✗          |
| ConvNeXt-XL   | 350.2M   | 10008     | 8823        | **11.84**   | ✗          |
| BLOOM-560M    | 559.2M   | 20005     | 18296       | **8.55**    | ✗          |
| BLOOM-560M    | 559.2M   | 5859      | 5544        | **5.38**    | ✓          |
| BLOOM-3B      | 3.0B     | 26472     | 24965       | **5.69**    | ✓          |
| BLOOM-7B      | 7.1B     | 48355     | 48184       | 0.35        | ✓          |
| Phi-1.5       | 1.4B     | 20098     | 19370       | **3.62**    | ✓          |
| Phi-2         | 2.8B     | 30301     | 28907       | **4.59**    | ✓          |
| Qwen-0.5B     | 464.0M   | 16437     | 15129       | **7.96**    | ✗          |
| Qwen-0.5B     | 464.0M   | 5509      | 5491        | 0.33        | ✓          |
| Qwen-1.8B     | 1.8B     | 14691     | 13673       | **6.93**    | ✓          |
| LLaMA-2-7B    | 6.7B     | 39115     | 35713       | **8.70**    | ✓          |
| Gemma-2B      | 2.5B     | 22118     | 20870       | **5.64**    | ✓          |
| Gemma-7B      | 8.5B     | 49424     | 48484       | 1.91        | ✓          |
| Vicuna-7B     | 6.7B     | 32351     | 28993       | **10.38**   | ✓          |
| ChatGLM3-6B   | 6.2B     | 37670     | 34614       | **8.11**    | ✓          |
| Falcon-7B     | 6.9B     | 40548     | 37099       | **8.51**    | ✓          |
### Memory Usage and Savings for Lion and Lion-R
| Model         | # Params | Lion (MB) | Lion-R (MB) | Savings (%) | ZeRO       |
|---------------|----------|-----------|-------------|-------------|------------|
| ViT-S         | 22.9M    | 415       | 327         | **21.21**   | ✗          |
| ViT-B         | 88.2M    | 1629      | 1231        | **24.45**   | ✗          |
| ViT-L         | 305.5M   | 5144      | 3827        | **25.60**   | ✗          |
| ViT-H         | 630.8M   | 10687     | 8087        | **24.33**   | ✗          |
| ViT-G         | 1.0B     | 17226     | 13189       | **23.43**   | ✗          |
| ConvNeXt-T    | 28.6M    | 552       | 489         | **11.41**   | ✗          |
| ConvNeXt-S    | 50.2M    | 958       | 791         | **17.51**   | ✗          |
| ConvNeXt-B    | 88.6M    | 1529      | 1281        | **16.19**   | ✗          |
| ConvNeXt-L    | 197.8M   | 3521      | 2861        | **18.77**   | ✗          |
| ConvNeXt-XL   | 350.2M   | 5862      | 4618        | **21.22**   | ✗          |
| BLOOM-560M    | 559.2M   | 13294     | 11996       | **9.76**    | ✗          |
| BLOOM-560M    | 559.2M   | 4513      | 4508        | 0.12        | ✓          |
| BLOOM-3B      | 3.0B     | 21957     | 20462       | **6.81**    | ✓          |
| BLOOM-7B      | 7.1B     | 41306     | 37761       | **8.58**    | ✓          |
| Phi-1.5       | 1.4B     | 17950     | 17273       | **3.77**    | ✓          |
| Phi-2         | 2.8B     | 26159     | 24809       | **5.15**    | ✓          |
| Qwen-0.5B     | 464.0M   | 10614     | 9666        | **8.93**    | ✗          |
| Qwen-0.5B     | 464.0M   | 4897      | 4855        | 0.86        | ✓          |
| Qwen-1.8B     | 1.8B     | 38986     | 31562       | **19.04**   | ✗          |
| Qwen-1.8B     | 1.8B     | 11913     | 10945       | **8.13**    | ✓          |
| LLaMA-2-7B    | 6.7B     | 29007     | 25618       | **11.68**   | ✓          |
| LLaMA-2-13B   | 13.0B    | 47297     | 39249       | **17.02**   | ✓          |
| Gemma-2B      | 2.5B     | 18347     | 17123       | **6.67**    | ✓          |
| Gemma-7B      | 8.5B     | 48279     | 39416       | **8.08**    | ✓          |
| Vicuna-7B     | 6.7B     | 28978     | 25596       | **11.67**   | ✓          |
| Vicuna-13B    | 13.0B    | 47596     | 39514       | **16.98**   | ✓          |
| ChatGLM3-6B   | 6.2B     | 28302     | 25180       | **11.03**   | ✓          |
| Falcon-7B     | 6.9B     | 30187     | 26719       | **11.49**   | ✓          |

### Equivalence to the Original Algorithms
The memory-reduced variants AdamW-R and Adan-R maintain exact identical training dynamics as their original counterparts when initialized with the same random seed, as indicated by Table 1 of our paper. While Lion-R introduces a minor change in the computational sequence due to variable substitution, it retains theoretical equivalence with the original Lion optimizer, with a minimal impact on the overall optimization outcomes.