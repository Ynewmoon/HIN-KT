# HIN-KT
We propose HIN-KT, a novel Heterogeneous Information Network (HIN) based pre-processing model to further enhance the performance of existing KT models in this paper. In HIN-KT, a HIN is firstly built to model the student-question-skill interactions, which are derived based on the learning interactions of all students in an ITS. After that, HIN-KT employs the interaction information in HIN to pre-train embeddings for each question and then adopts the pre-trained embeddings to enhance the performance of deep KT models. 

Experimental results over several public KT datasets demonstrate that large gains on knowledge tracing can be achieved when the proposed HIN-KT is used to pre-train question embeddings via the information of student-question-skill interactions, followed by training the state-of-the-art deep KT models on the obtained question embeddings. 

In particular, the adoption of the HIN-KT pre-training model successfully improves the performance of the state-of-the-art deep KT models, i.e., DKT and CKT, averagely by $8.78\%$, which is a significant progress made in the knowledge tracing domain.


# Datasets
If you want to process the datasets by yourself, you can download them by the following links:

## ASSIST2009
https://sites.google.com/site/assistmentsdata/home/2009-2010-assistment-data/skill-builder-data-2009-2010
## Ednet
https://github.com/riiid/ednet
## Statics2011
https://pslcdatashop.web.cmu.edu/DatasetInfo?datasetId=507
