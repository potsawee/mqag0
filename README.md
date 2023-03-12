MQAG: Multiple-choice Question Answering and Generation for Assessing Information Consistency
============================================================
Code for our paper [MQAG: Multiple-choice Question Answering and Generation for Assessing Information Consistency in Summarization](https://arxiv.org/abs/2301.12307)


- Please read our paper for the information on MQAG
- Model weights are available:
	- HuggingFace:
		- ```generation_model_1```: https://huggingface.co/potsawee/t5-large-generation-race-QuestionAnswer
		- ```generation_model_2```: https://huggingface.co/potsawee/t5-large-generation-race-Distractor
		- ```answering_model```: https://huggingface.co/potsawee/longformer-large-4096-answering-race

	- Google Drive:
		- ```generation_model_1```: [t5-large-generation-Race-QuestionAnswer](https://drive.google.com/file/d/1FSnwgqSFZ6wcVco78DO_aXj5D0b0LK1e/view?usp=share_link)
		- ```generation_model_2```: [t5-large-generation-Race-Distractor](https://drive.google.com/file/d/1zFcps700Vhjzt8m8jxhq6XRVQUVk95pw/view?usp=share_link)
		- ```answering_model```: [longformer-large-4096-Race-Answering](https://drive.google.com/file/d/1bToo1l6zd934uLhsvLY5Am0dFaKDS-Ph/view?usp=share_link)
- Requirements: We've tested on python 3.8 and other packages are shown in ```requirements.txt```

Running MQAG Inference
------------------------------------------------------------
```python inference_mqag.py``` takes the arguments below. source and summary files contain text in the one document per line format (see examples/...)

- **source\_path**: path to source (text) file
- **summary\_path**: path to summary (text) file
- **mqag\_variant**: mqag_src | mqag_sum
- **num\_samples**: number of questions to be drawn
- **generation\_model1\_path**: path to Question+Answer Gen (t5-large)
- **generation\_model2\_path**: path to Distractor Gen (t5-large)
- **generation\_model\_type**: e.g. t5-large
- **answering\_model\_path**: path to Answering model (longformer)
- **answering\_model\_type**: e.g. longformer
- **use\_gpu**: True | False (whether or not to use GPU if available)
- **verbose**: True | False (whether or not to print information)

Example usage:

	python inference_mqag.py \
	    --source_path=examples/0_source.txt \
	    --summary_path=examples/0_summary.txt \
	    --mqag_variant=mqag_sum \
	    --num_samples=10 \
	    --generation_model1_path=model_weights/t5-large-generation-Race-QuestionAnswer.pt \
	    --generation_model2_path=model_weights/t5-large-generation-Race-Distractor.pt \
	    --answering_model_path=model_weights/longformer-large-4096-Race-Answering.pt \
	    --use_gpu=True \
	    --verbose=True

Example Output:

	[document=1/2, multiple-choice question=1/10]
	Question: Two security guards have been threatened during a robbery at a _.
	(1) bank
	(2) securityguard
	(3) school
	(4) mosque
	prob_sum_y = 0.995091	0.004669	0.000126	0.000114
	prob_doc_x = 0.981052	0.017671	0.000337	0.000941
	prob_nocontext = 0.716027	0.043105	0.061257	0.179611

Experimental Results
------------------------------------------------------------
The pearson correlation coefficient (PCC) between the MQAG-score and human judgements are shown below.
|   Method  | QAG-CNNDM | QAG-XSum | XSum-Faithful | XSum-Factual | Podcast | SummEval-Rel | SummEval-Cons |
|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|
| MQAG-Sum (KL-div) | 0.461 | 0.283  | 0.135 | 0.179 | 0.761 | 0.772 | 0.942 |
| MQAG-Src (KL-div) | 0.144 | 0.097  | 0.087 | 0.054 | 0.320 | 0.559 | 0.599 |

**\*Update (06/02/2023)**: We found that comparing two distributions using a *bounded* distance such as [total variation](https://en.wikipedia.org/wiki/Total_variation_distance_of_probability_measures) yields better results than KL-diveregence. Additional statistical distances will added to the MQAG repository soon.

Training Multiple-choice QG and QA models
------------------------------------------------------------
Note that trained weights are available with the download links provided at the beginning of README. Here, we provide the scripts to train QG and QA models, or fine-tune to other multiple-choice datasets. Hyparameters and configurations are set manually inside the scripts just before ```def experiment()```. The current version only supports T5 and Longformer, but you're welcome to modify the code to use a different architecture.

### QG system
There are two generation models: (1) Question + Answer(supposedly correct answer) Generation; (2) Distrator Generation (the remaining options in addition to the answer).

- **model1**: Question + Answer Generation Model

		python train_generation_qa.py

- **model2**: Distrator Generation Model

		python train_generation_distractors.py

### QA system

- one answering model for predicting probablity over the options

		python train_answering.py


Links to Datasets
------------------------------------------------------------
We refer to the original papers that release the dataset.

- **Multiple-choice Reading Comprehension**
	- [RACE](https://www.cs.cmu.edu/~glai1/data/race/)

- **Summary Evaluation**
	- [QAG](https://github.com/W4ngatang/qags)
	- [XSum-Faithful](https://github.com/google-research-datasets/xsum_hallucination_annotations)
	- [Podcast Assessment](https://github.com/potsawee/podcast_summary_assessment)
	- [SummEval](https://github.com/Yale-LILY/SummEval)

Citation
-----------------------------------------

	@article{manakul2023mqag,
	  title={MQAG: Multiple-choice Question Answering and Generation for Assessing Information Consistency in Summarization},
	  author={Manakul, Potsawee and Liusie, Adian and Gales, Mark JF},
	  journal={arXiv preprint arXiv:2301.12307},
	  year={2023}
	}
