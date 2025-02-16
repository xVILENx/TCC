import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification, Trainer, TrainingArguments, DataCollatorForTokenClassification, TrainerCallback
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import re
import os
import matplotlib.pyplot as plt
from collections import Counter

os.environ["WANDB_DISABLED"] = "true"

# --- Parte 1: Configuração do modelo de NER ---

# Usaremos o BioBERT para token classification (NER)
# Note que alteramos para AutoModelForTokenClassification e definimos 3 rótulos: "O", "B-GENE" e "B-DISEASE"
label2id = {"O": 0, "B-GENE": 1, "B-DISEASE": 2}
id2label = {v: k for k, v in label2id.items()}

tokenizer = AutoTokenizer.from_pretrained("dmis-lab/biobert-base-cased-v1.1")
model = AutoModelForTokenClassification.from_pretrained(
    "dmis-lab/biobert-base-cased-v1.1",
    num_labels=len(label2id),
    id2label=id2label,
    label2id=label2id
)

# --- Parte 2: Criação do dataset anotado por supervisão distante ---

# Função para carregar a lista de genes a partir de um arquivo
def load_gene_list(file_path):
    with open(file_path, 'r') as f:
        genes = [line.strip() for line in f if line.strip()]
    return genes

# Carregar a lista de genes (não altere sua lista)
gene_list = load_gene_list("/content/gene_id_list.13")
# Converter para um conjunto com todos os termos em maiúsculas para comparação
valid_genes = {gene.upper() for gene in gene_list}
print(f"Genes carregados: {valid_genes}")

# Função para normalizar um token extraído:
def normalize_gene(token):
    # Remove hífens (por exemplo, "IL-17A" → "IL17A")
    token = token.replace("-", "")
    # Substitui letras gregas (por exemplo, "γ" ou "Γ" → "G")
    token = token.replace("γ", "G").replace("Γ", "G")
    return token.upper()

# Função para normalizar as doenças extraídas
def normalize_disease(d):
    if re.search(r'Alzheimer', d, re.IGNORECASE):
        return "ALZHEIMER'S DISEASE"
    elif re.search(r'COVID-19|SARS[-\s]?COV[-\s]?2', d, re.IGNORECASE):
        return "COVID-19"
    return d.upper()

# Função para gerar rótulos BIO para um texto usando a lista de genes e padrões para doenças.
def get_bio_labels(text, valid_genes):
    words = text.split()
    labels = []
    # Usaremos um padrão para doenças para identificar tokens que representem COVID ou Alzheimer
    for word in words:
        if sum(1 for c in word if c.isupper()) < 2:
            labels.append("O")
            continue
        norm = normalize_gene(word)
        if norm in valid_genes:
            labels.append("B-GENE")
        elif re.fullmatch(r'\b(COVID[- ]?19|SARS[- ]?COV[- ]?2)\b', word, re.IGNORECASE):
            labels.append("B-DISEASE")
        elif re.fullmatch(r'\b(Alzheimer(?:\'s)?(?:\s*disease)?)\b', word, re.IGNORECASE):
            labels.append("B-DISEASE")
        else:
            labels.append("O")
    return words, labels

# Textos de treinamento
train_texts = [
    # 1
    "Introduction: The COVID-19 pandemic represented one of the most significant challenges to researchers and healthcare providers. Several factors determine the disease severity, whereas none alone can explain the tremendous variability. The Single nucleotide variants (SNVs) in angiotensin-converting enzyme-2 (ACE2) and transmembrane serine protease type-2 (TMPRSS2) genes affect the virus entry and are considered possible risk factors for COVID-19. Methods: We compiled a panel of gene variants from both genes and used in-silico analysis to predict their significance. We performed biological validation to assess their capacity to alter the ACE2 interaction with the virus spike protein. Subsequently, we conducted a retrospective comparative genome analysis on those variants in the Emirati patients with different disease severity (total of 96) along with 69 healthy control subjects. Results: Our results showed that the Emirati population lacks the variants that were previously reported as associated with disease severity, whereas a new variant in ACE2 \"Chr X:g.15584534\" was associated with disease severity specifically among female patients. In-silico analysis revealed that the new variant can determine the ACE2 gene transcription. Several cytokines (GM-CSF and IL-6) and chemokines (MCP-1/CCL2, IL-8/CXCL8, and IP-10/CXCL10) were markedly increased in COVID-19 patients with a significant correlation with disease severity. The newly reported genetic variant of ACE2 showed a positive correlation with CD40L, IL-1β, IL-2, IL-15, and IL-17A in COVID-19 patients. Conclusion: Whereas COVID-19 represents now a past pandemic, our study underscores the importance of genetic factors specific to a population, which can influence both the susceptibility to viral infections and the level of severity; subsequently expected required preparedness in different areas of the world.",
    # 2
    "Background: There is growing evidence of a strong relationship between COVID-19 and myocarditis. However, there are few bioinformatics-based analyses of critical genes and the mechanisms related to COVID-19 Myocarditis. This study aimed to identify critical genes related to COVID-19 Myocarditis by bioinformatic methods, explore the biological mechanisms and gene regulatory networks, and probe related drugs. Methods: The gene expression data of GSE150392 and GSE167028 were obtained from the Gene Expression Omnibus (GEO), including cardiomyocytes derived from human induced pluripotent stem cells infected with SARS-CoV-2 in vitro and GSE150392 from patients with myocarditis infected with SARS-CoV-2 and the GSE167028 gene expression dataset. Differentially expressed genes (DEGs) (adjusted P-Value <0.01 and |Log2 Fold Change| ≥2) in GSE150392 were assessed by NetworkAnalyst 3.0. Meanwhile, significant modular genes in GSE167028 were identified by weighted gene correlation network analysis (WGCNA) and overlapped with DEGs to obtain common genes. Functional enrichment analyses were performed by using the \"clusterProfiler\" package in the R software, and protein-protein interaction (PPI) networks were constructed on the STRING website (https://cn.string-db.org/). Critical genes were identified by the CytoHubba plugin of Cytoscape by 5 algorithms. Transcription factor-gene (TF-gene) and Transcription factor-microRibonucleic acid (TF-miRNA) coregulatory networks construction were performed by NetworkAnalyst 3.0 and displayed in Cytoscape. Finally, Drug Signatures Database (DSigDB) was used to probe drugs associated with COVID-19 Myocarditis. Results: Totally 850 DEGs (including 449 up-regulated and 401 down-regulated genes) and 159 significant genes in turquoise modules were identified from GSE150392 and GSE167028, respectively. Functional enrichment analysis indicated that common genes were mainly enriched in biological processes such as cell cycle and ubiquitin-protein hydrolysis. 6 genes (CDK1, KIF20A, PBK, KIF2C, CDC20, UBE2C) were identified as critical genes. TF-gene interactions and TF-miRNA coregulatory network were constructed successfully. A total of 10 drugs, (such as Etoposide, Methotrexate, Troglitazone, etc) were considered as target drugs for COVID-19 Myocarditis. Conclusions: Through bioinformatics method analysis, this study provides a new perspective to explore the pathogenesis, gene regulatory networks and provide drug compounds as a reference for COVID-19 Myocarditis. It is worth highlighting that critical genes (CDK1, KIF20A, PBK, KIF2C, CDC20, UBE2C) may be potential biomarkers and treatment targets of COVID-19 Myocarditis for future study.",
    # 3
    "Severe acute respiratory syndrome coronavirus 2 (SARS-CoV-2) is a known virus that leads to a respiratory disease called coronavirus disease 19 (COVID-19). Natural killer (NK) cells, as members of innate immunity, possess crucial roles in restricting viral infections, including COVID-19. Their functions and development depend on receiving signals through various receptors, of which killer cell immunoglobulin-like receptors (KIRs) belong to the most effective ones. Different studies investigated the association between KIR gene content and susceptibility to COVID-19. Since previous studies have yielded contradictory results, we designed this meta-analysis study to draw comprehensive conclusions about COVID-19 risk and KIR gene association. According to PRISMA guidelines, a systematic search was performed in the electronic databases to find all studies investigating KIR gene contents in COVID-19 patients before March 2023. Any association between KIR genes and COVID-19 risk was determined by calculating pooled odds ratio (OR) and 95% confidence interval (CI). After applying the inclusion and exclusion criteria, 1673 COVID-19 patients and 1526 healthy controls from eight studies were included in this meta-analysis. As the main results, we observed a positive association between the 2DL3 (OR = 1.48, 95% CI = 1.17-1.88, P < 0.001) and susceptibility to COVID-19 and a negative association between the 2DP1 and the risk for COVID-19 (OR = 0.48, 95% CI = 0.23-0.99, P = 0.049). This meta-analysis demonstrated that KIR2DL3, as a member of iKIRs, might be associated with an increased risk of COVID-19 disease.",
    # 4
    "Objectives: Host genetic factors contribute to the variable severity of COVID-19. We examined genetic variants from genome-wide association studies and candidate gene association studies in a cohort of patients with COVID-19 and investigated the role of early SARS-CoV-2 strains in COVID-19 severity. Methods: This case-control study included 123 COVID-19 cases (hospitalized or ambulatory) and healthy controls from the state of Baden-Wuerttemberg, Germany. We genotyped 30 single nucleotide polymorphisms, using a custom-designed panel. Cases were also compared with the 1000 genomes project. Polygenic risk scores were constructed. SARS-CoV-2 genomes from 26 patients with COVID-19 were sequenced and compared between ambulatory and hospitalized cases, and phylogeny was reconstructed. Results: Eight variants reached nominal significance and two were significantly associated with at least one of the phenotypes \"susceptibility to infection\", \"hospitalization\", or \"severity\": rs73064425 in LZTFL1 (hospitalization and severity, P <0.001) and rs1024611 near CCL2 (susceptibility, including 1000 genomes project, P = 0.001). The polygenic risk score could predict hospitalization. Most (23/26, 89%) of the SARS-CoV-2 genomes were classified as B.1 lineage. No associations of SARS-CoV-2 mutations ou lineages with severity were observed. Conclusion: These host genetic markers provide insights into pathogenesis and enable risk classification. Variants which reached nominal significance should be included in larger studies.",
    # 5
    "The novel coronavirus SARS-CoV-2 is damaging the world's social and economic fabrics seriously. Effective drugs are urgently needed to decrease the high mortality rate of COVID-19 patients. Unfortunately, effective antiviral drugs or vaccines are currently unavailable. Herein, we systematically evaluated the effect of SARS-CoV-2 on gene expression of both lung tissue and blood from COVID-19 patients using transcriptome profiling. Differential gene expression analysis revealed potential core mechanism of COVID-19-induced pneumonia in which IFN-α, IFN-β, IFN-γ, TNF and IL6 triggered cytokine storm mediated by neutrophil, macrophage, B and DC cells. Weighted gene correlation network analysis identified two gene modules that are highly correlated with clinical traits of COVID-19 patients, and confirmed that over-activation of immune system-mediated cytokine release syndrome is the underlying pathogenic mechanism for acute phase of COVID-19 infection. It suggested that anti-inflammatory therapies may be promising regimens for COVID-19 patients. Furthermore, drug repurposing analysis of thousands of drugs revealed that TNFα inhibitor etanercept and γ-aminobutyric acid-B receptor (GABABR) agonist baclofen showed most significant reversal power to COVID-19 gene signature, so we are highly optimistic about their clinical use for COVID-19 treatment. In addition, our results suggested that adalimumab, tocilizumab, rituximab and glucocorticoids may also have beneficial effects in restoring normal transcriptome, but not chloroquine, hydroxychloroquine or interferons. Controlled clinical trials of these candidate drugs are needed in search of effective COVID-19 treatment in current crisis.",
    # 6
    "Cell entry of SARS-CoV-2, the novel coronavirus causing COVID-19, is facilitated by host cell angiotensin-converting enzyme 2 (ACE2) and transmembrane serine protease 2 (TMPRSS2). We aimed to identify and characterize genes that are co-expressed with ACE2 and TMPRSS2, and to further explore their biological functions and potential as druggable targets. Using the gene expression profiles of 1,038 lung tissue samples, we performed a weighted gene correlation network analysis (WGCNA) to identify modules of co-expressed genes. We explored the biology of co-expressed genes using bioinformatics databases, and identified known drug-gene interactions. ACE2 was in a module of 681 co-expressed genes; 10 genes with moderate-high correlation with ACE2 (r > 0.3, FDR < 0.05) had known interactions with existing drug compounds. TMPRSS2 was in a module of 1,086 co-expressed genes; 31 of these genes were enriched in the gene ontology biologic process 'receptor-mediated endocytosis', and 52 TMPRSS2-correlated genes had known interactions with drug compounds. Dozens of genes are co-expressed with ACE2 and TMPRSS2, many of which have plausible links to COVID-19 pathophysiology. Many of the co-expressed genes are potentially targetable with existing drugs, which may accelerate the development of COVID-19 therapeutics.",
    # 7
    "Severe acute respiratory syndrome coronavirus-2 (SARS-CoV-2) infection results in the development of a highly contagious respiratory ailment known as new coronavirus disease (COVID-19). Despite the fact that the prevalence of COVID-19 continues to rise, it is still unclear how people become infected with SARS-CoV-2 and how patients with COVID-19 become so unwell. Detecting biomarkers for COVID-19 using peripheral blood mononuclear cells (PBMCs) may aid in drug development and treatment. This research aimed to find blood cell transcripts that represent levels of gene expression associated with COVID-19 progression. Through the development of a bioinformatics pipeline, two RNA-Seq transcriptomic datasets and one microarray dataset were studied and discovered 102 significant differentially expressed genes (DEGs) that were shared by three datasets derived from PBMCs. To identify the roles of these DEGs, we discovered disease-gene association networks and signaling pathways, as well as we performed gene ontology (GO) studies and identified hub protein. Identified significant gene ontology and molecular pathways improved our understanding of the pathophysiology of COVID-19, and our identified blood-based hub proteins TPX2, DLGAP5, NCAPG, CCNB1, KIF11, HJURP, AURKB, BUB1B, TTK, and TOP2A could be used for the development of therapeutic intervention. In COVID-19 subjects, we discovered effective putative connections between pathological processes in the transcripts blood cells, suggesting that blood cells could be used to diagnose and monitor the disease's initiation and progression as well as developing drug therapeutics.",
    # 8
    "Protection against severe acute respiratory syndrome coronavirus 2 (SARS-CoV-2) infection and associated clinical sequelae requires well-coordinated metabolic and immune responses that limit viral spread and promote recovery of damaged systems. However, the role of the gut microbiota in regulating these responses has not been thoroughly investigated. In order to identify mechanisms underpinning microbiota interactions with host immune and metabolic systems that influence coronavirus disease 2019 (COVID-19) outcomes, we performed a multi-omics analysis on hospitalized COVID-19 patients and compared those with the most severe outcome (i.e. death, n = 41) to those with severe non-fatal disease (n = 89), or mild/moderate disease (n = 42), that recovered. A distinct subset of 8 cytokines (e.g. TSLP) and 140 metabolites (e.g. quinolinate) in sera identified those with a fatal outcome to infection. In addition, elevated levels of multiple pathobionts and lower levels of protective or anti-inflammatory microbes were observed in the fecal microbiome of those with the poorest clinical outcomes. Weighted gene correlation network analysis (WGCNA) identified modules that associated severity-associated cytokines with tryptophan metabolism, coagulation-linked fibrinopeptides, and bile acids with multiple pathobionts, such as Enterococcus. In contrast, less severe clinical outcomes are associated with clusters of anti-inflammatory microbes such as Bifidobacterium or Ruminococcus, short chain fatty acids (SCFAs) and IL-17A. Our study uncovered distinct mechanistic modules that link host and microbiome processes with fatal outcomes to SARS-CoV-2 infection. These features may be useful to identify at risk individuals, but also highlight a role for the microbiome in modifying hyperinflammatory responses to SARS-CoV-2 and other infectious agents.",
    # 9
    "The Severe acute respiratory syndrome may be caused by coronavirus disease which has resulted in a global pandemic. Polymorphisms in the population play a role in susceptibility to severity. We aimed to perform a systematic review related to the effect of single nucleotide polymorphisms in the development of severe acute respiratory syndrome (SARS). Twenty-eight eligible articles published were identified in PubMed, ScienceDirect, Web of Science, PMC Central and Portal BVS and additional records, with 20 studies performed in China. Information on study characteristics, genetic polymorphisms, and comorbidities was extracted. Study quality was assessed by the STrengthening the REporting of Genetic Association (STREGA) guideline. Few studies investigated the presence of polymorphisms in HLA, ACE1, OAS-1, MxA, PKR, MBL, E-CR1, FcγRIIA, MBL2, L-SIGN (CLEC4M), IFNG, CD14, ICAM3, RANTES, IL-12 RB1, TNFA, CXCL10/IP-10, CD209 (DC-SIGN), AHSG, CYP4F3 and CCL2 with the susceptibility or protection to SARS-Cov. This review provides comprehensive evidence of the association between genetic polymorphisms and susceptibility or protection to severity SARS-Cov. The literature about coronavirus infection, susceptibility to severe acute respiratory syndrome (SARS) and genetic variations is scarce. Further studies are necessary to provide more concrete evidence, mainly related to Covid-19.",
    # 10
    "SARS-CoV-2 causes substantial extrapulmonary manifestations in addition to pulmonary disease. Some of the major organs affected are cardiovascular, hematological and thrombotic, renal, neurological, and digestive systems. These types of muti-organ dysfunctions make it difficult and challenging for clinicians to manage and treat COVID-19 patients. The article focuses to identify potential protein biomarkers that can flag various organ systems affected in COVID-19. Publicly reposited high throughput proteomic data from human serum (HS), HEK293T/17 (HEK) and Vero E6 (VE) kidney cell culture were downloaded from ProteomeXchange consortium. The raw data was analyzed in Proteome Discoverer 2.4 to delineate the complete list of proteins in the three studies. These proteins were analyzed in Ingenuity Pathway Analysis (IPA) to associate them to various organ diseases. The shortlisted proteins were analyzed in MetaboAnalyst 5.0 to shortlist potential biomarker proteins. These were then assessed for disease-gene association in DisGeNET and validated by Protein-protein interactome (PPI) and functional enrichment studies (GO_BP, KEGG and Reactome pathways) in STRING. Protein profiling resulted in shortlisting 20 proteins in 7 organ systems. Of these 15 proteins showed at least 1.25-fold changes with a sensitivity and specificity of 70%. Association analysis further shortlisted 10 proteins with a potential association with 4 organ diseases. Validation studies established possible interacting networks and pathways affected, confirmingh the ability of 6 of these proteins to flag 4 different organ systems affected in COVID-19 disease. This study helps to establish a platform to seek protein signatures in different clinical phenotypes of COVID-19. The potential biomarker candidates that can flag organ systems involved are: (a) Vitamin K-dependent protein S and Antithrombin-III for hematological disorders; (b) Voltage-dependent anion-selective channel protein 1 for neurological disorders; (c) Filamin-A for cardiovascular disorder and, (d) Peptidyl-prolyl cis-trans isomerase A and Peptidyl-prolyl cis-trans isomerase FKBP1A for digestive disorders.",
    # 11
    "Messenger RNA (mRNA) vaccines represent a new class of vaccines that has been shown to be highly effective during the COVID-19 pandemic and that holds great potential for other preventative and therapeutic applications. While it is known that the transcriptional activity of various genes is altered following mRNA vaccination, identifying and studying gene networks could reveal important scientific insights that might inform future vaccine designs. In this study, we conducted an in-depth weighted gene correlation network analysis of the blood transcriptome before and 24 h after the second and third vaccination with licensed mRNA vaccines against COVID-19 in humans, following a prime vaccination with either mRNA or ChAdOx1 vaccines. Utilizing this unsupervised gene network analysis approach, we identified distinct modular networks of co-varying genes characterized by either an expressional up- or downregulation in response to vaccination. Downregulated networks were associated with cell metabolic processes and regulation of transcription factors, while upregulated networks were associated with myeloid differentiation, antigen presentation, and antiviral, interferon-driven pathways. Within this interferon-associated network, we identified highly connected hub genes such as STAT2 and RIGI and associated upstream transcription factors, potentially playing important regulatory roles in the vaccine-induced immune response. The expression profile of this network significantly correlated with S1-specific IgG levels at the follow-up visit in vaccinated individuals. Those findings could be corroborated in a second, independent cohort of mRNA vaccine recipients. Collectively, results from this modular gene network analysis enhance the understanding of mRNA vaccines from a systems immunology perspective. Influencing specific gene networks could lead to optimized vaccines that elicit augmented vaccine responses.",
    # 12
    "The immune system and neuroinflammation are now well established in the aetiology of neurodegeneration. Previous studies of transcriptomic and gene association studies have highlighted the potential of the 2'-5' oligoadenylate synthetase 1 (OAS1) to play a role in Alzheimer's disease. OAS1 is a viral response gene, interferon-induced, dsRNA activated enzyme, which binds RNase L to degrade dsRNA, and has been associated with COVID-19 response. This study explores whether a viral defence gene could play a vital role in neurodegeneration pathology. The genotyping of five SNPs across the OAS1 locus was conducted in the Brains for Dementia Research (BDR) Cohort for association with AD. RNA-sequencing data were explored for differences in OAS1 gene expression between phenotypes and genotypes. Finally, levels of dsRNA were measured in control cell lines, prior to and after exposure to amyloid oligomers and in cells harbouring a dementia-relevant mutation. No association of any of the OAS1 SNPs investigated were associated with the AD phenotype in the BDR cohort. However, gene expression data supported the previous observation that the minor allele haplotype was associated with higher levels of the OAS1 gene expression and the presence of an alternative transcript. Further to this, the presence of endogenous dsRNA was found to increase with exposure to amyloid oligomers and in the cell line with a dementia-relevant mutation. The data presented here suggest further exploration of the OAS1 gene in relation to dementia is warranted. Investigations of whether carriers of the protective OAS1 haplotype lower dsRNA presence and in turn lower inflammation and cell death are required to support the role of the gene as a moderator of neurodegeneration.",
    # 13
    "Amyloid plaques, mainly composed of abnormally aggregated amyloid β-protein (Aβ) in the brain parenchyma, and neurofibrillary tangles (NFTs), consisting of hyperphosphorylated tau protein aggregates in neurons, are two pathological hallmarks of Alzheimer's disease (AD). Aβ fibrils and tau aggregates in the brain are closely associated with neuroinflammation and synapse loss, characterized by activated microglia and dystrophic neurites. Genome-wide genetic association studies revealed important roles of innate immune cells in the pathogenesis of late-onset AD by recognizing a dozen genetic risk loci that modulate innate immune activities. Furthermore, microglia, brain resident innate immune cells, have been increasingly recognized to play key, opposing roles in AD pathogenesis by either eliminating toxic Aβ aggregates and enhancing neuronal plasticity or producing proinflammatory cytokines, reactive oxygen species, and synaptotoxicity. Aggregated Aβ binds to toll-like receptor 4 (TLR4) and activates microglia, resulting in increased phagocytosis and cytokine production. Complement components are associated with amyloid plaques and NFTs. Aggregated Aβ can activate complement, leading to synapse pruning and loss by microglial phagocytosis. Systemic inflammation can activate microglial TLR4, NLRP3 inflammasome, and complement in the brain, leading to neuroinflammation, Aβ accumulation, synapse loss and neurodegeneration. The host immune response has been shown to function through complex crosstalk between the TLR, complement and inflammasome signaling pathways. Accordingly, targeting the molecular mechanisms underlying the TLR-complement-NLRP3 inflammasome signaling pathways can be a preventive and therapeutic approach for AD.",
    # 14
    "Aim: Heredity plays an important role in the pathogenesis of Alzheimer's disease (AD) especially for single-nucleotide polymorphism (SNPs) of susceptible genes, which is one of the significant factors in the pathogenesis of AD. The SNPs of BIN1 rs744373, BIN1 rs7561528 and GAB2 rs2373115 are associated with AD in Asian and white people. Methods: We included 34 studies with a total of 38 291 patients with AD and 55 538 controls of diverse races from four main databases. We used meta-analysis to obtain I2 -values and odds ratios of five genetic models in three SNPs. We carried out analysis of sensitivity, subgroup, publication bias and linkage disequilibrium test Results: The forest plots showed the odds ratio value of the three SNPs was >1 in white individuals, but not Asian individuals, in their genetic model. The funnel plot was symmetrical, and the D'-value was 0.986 between rs744373 and rs7561528. Conclusions: BIN1 rs744373, BIN1 rs7561528 and GAB2 rs2373115 are pathogenicity sites for AD in white people, and also rs7561528 belongs to a risk site in Asian people. The rs7561528 and rs744373 SNPs have strong linkage disequilibrium in Chinese people. In addition, apolipoprotein E ε4 status promotes them to result in the pathogenesis of AD. Geriatr Gerontol Int 2021; 21: 185-191.",
    # 15
    "Alzheimer's disease (AD) is commonly considered as the most prominent dementing disorder globally and is characterized by the deposition of misfolded amyloid-β (Aβ) peptide and the aggregation of neurofibrillary tangles. Immunological disturbances and neuroinflammation, which result from abnormal immunological reactivations, are believed to be the primary stimulating factors triggering AD-like neuropathy. It has been suggested by multiple previous studies that a bunch of AD key influencing factors might be attributed to genes encoding human leukocyte antigen (HLA), whose variety is an essential part of human adaptive immunity. A wide range of activities involved in immune responses may be determined by HLA genes, including inflammation mediated by the immune response, T-cell transendothelial migration, infection, brain development and plasticity in AD pathogenesis, and so on. The goal of this article is to review the recent epidemiological findings of HLA (mainly HLA class I and II) associated with AD and investigate to what extent the genetic variations of HLA were clinically significant as pathogenic factors for AD. Depending on the degree of contribution of HLA in AD pathogenesis, targeted research towards HLA may propel AD therapeutic strategies into a new era of development.",
    # 16
    "Microglia are resident myeloid cells in the central nervous system (CNS) with a unique developmental origin, playing essential roles in developing and maintaining the CNS environment. Recent studies have revealed the involvement of microglia in neurodegenerative diseases, such as Alzheimer's disease, through the modulation of neuroinflammation. Several members of the Siglec family of sialic acid recognition proteins are expressed on microglia. Since the discovery of the genetic association between a polymorphism in the CD33 gene and late-onset Alzheimer's disease, significant efforts have been made to elucidate the molecular mechanism underlying the association between the polymorphism and Alzheimer's disease. Furthermore, recent studies have revealed additional potential associations between Siglecs and Alzheimer's disease, implying that the reduced signal from inhibitory Siglec may have an overall protective effect in lowering the disease risk. Evidences suggesting the involvement of Siglecs in other neurodegenerative diseases are also emerging. These findings could help us predict the roles of Siglecs in other neurodegenerative diseases. However, little is known about the functionally relevant Siglec ligands in the brain, which represents a new frontier. Understanding how microglial Siglecs and their ligands in CNS contribute to the regulation of CNS homeostasis and pathogenesis of neurodegenerative diseases may provide us with a new avenue for disease prevention and intervention.",
    # 17
    "Transthyretin (TTR) is secreted by hepatocytes, retinal pigment epithelial cells, pancreatic α and β cells, choroid plexus epithelium, and neurons under stress. The choroid plexus product is the main transporter of the thyroid hormone thyroxine (T4) to the brain during early development. TTR is one of three relatively abundant cerebrospinal fluid (CSF) proteins (Apolipoprotein J [ApoJ] (also known as clusterin), Apolipoprotein E [ApoE], and TTR) that interact with Aβ peptides in vitro, in some instances inhibiting their aggregation and toxicity. It is now clear that clusterin functions as an extracellular, and perhaps intracellular, chaperone for many misfolded proteins and that variation in its gene (Clu) is associated with susceptibility to sporadic Alzheimer's disease (AD). The function of ApoE in AD is not yet completely understood, although the ApoE4 allele has the strongest genetic association with the development of sporadic late onset AD. Despite in vitro and in vivo evidence of the interaction between TTR and Aβ, genomewide association studies including large numbers of sporadic Alzheimer's disease patients have failed to show significant association between variation in the TTR gene and disease prevalence. Early clinical studies suggested an inverse relationship between CSF TTR levels and AD and the possibility of using the reduced CSF TTR concentration as a biomarker. Later, more extensive analyses indicated that CSF TTR concentrations may be increased in some patients with AD. While the observed changes in TTR may be pathogenetically or biologically interesting because of the inconsistency and lack of specificity, they offered no benefit diagnostically or prognostically either independently or when added to currently employed CSF biomarkers, i.e., decreased Aβ1-42 and increased Tau and phospho-Tau. While some clinical data suggest that increases in CSF TTR may occur early in the disease with a significant decrease late in the course, without additional, more granular data, CSF TTR changes are neither consistent nor specific enough to warrant their use as a specific AD biomarker.",
    # 18
    "Human apolipoprotein E (ApoE) was first identified as a polymorphic gene in the 1970s; however, the genetic association of ApoE genotypes with late-onset sporadic Alzheimer's disease (sAD) was only discovered 20 years later. Since then, intensive research has been undertaken to understand the molecular effects of ApoE in the development of sAD. Despite three decades' worth of effort and over 10,000 papers published, the greatest mystery in the ApoE field remains: human ApoE isoforms differ by only one or two amino acid residues; what is responsible for their significantly distinct roles in the etiology of sAD, with ApoE4 conferring the greatest genetic risk for sAD whereas ApoE2 providing exceptional neuroprotection against sAD. Emerging research starts to point to a novel and compelling hypothesis that the sialoglycans posttranslationally appended to human ApoE may serve as a critical structural modifier that alters the biology of ApoE, leading to the opposing impacts of ApoE isoforms on sAD and likely in the peripheral systems as well. ApoE has been shown to be posttranslationally glycosylated in a species-, tissue-, and cell-specific manner. Human ApoE, particularly in brain tissue and cerebrospinal fluid (CSF), is highly glycosylated, and the glycan chains are exclusively attached via an O-linkage to serine or threonine residues. Moreover, studies have indicated that human ApoE glycans undergo sialic acid modification or sialylation, a structural alteration found to be more prominent in ApoE derived from the brain and CSF than plasma. However, whether the sialylation modification of human ApoE has a biological role is largely unexplored. Our group recently first reported that the three major isoforms of human ApoE in the brain undergo varying degrees of sialylation, with ApoE2 exhibiting the most abundant sialic acid modification, whereas ApoE4 is the least sialylated. Our findings further indicate that the sialic acid moiety on human ApoE glycans may serve as a critical modulator of the interaction of ApoE with amyloid β (Aβ) and downstream Aβ pathogenesis, a prominent pathologic feature in AD. In this review, we seek to provide a comprehensive summary of this exciting and rapidly evolving area of ApoE research, including the current state of knowledge and opportunities for future exploration."
]
# Gerar rótulos automaticamente para cada texto
train_data = []  # Cada item será uma tupla (words, labels)
for text in train_texts:
    words, labels = get_bio_labels(text, valid_genes)
    train_data.append((words, labels))
    print("Texto:", text)
    print("Palavras:", words)
    print("Rótulos:", labels)
    print("-----")

# Função para tokenizar e alinhar os rótulos com os subwords do tokenizador
def tokenize_and_align_labels(words, labels):
    tokenized_inputs = tokenizer(words, is_split_into_words=True, truncation=True, padding="max_length", max_length=512, return_offsets_mapping=True)
    word_ids = tokenized_inputs.word_ids()
    aligned_labels = []
    previous_word_idx = None
    for word_idx in word_ids:
        if word_idx is None:
            aligned_labels.append(-100)
        elif word_idx != previous_word_idx:
            aligned_labels.append(label2id[labels[word_idx]])
        else:
            aligned_labels.append(label2id[labels[word_idx]])
        previous_word_idx = word_idx
    tokenized_inputs["labels"] = aligned_labels
    tokenized_inputs.pop("offset_mapping")
    return tokenized_inputs

# Tokenizar e alinhar os rótulos para todo o dataset de treinamento
tokenized_train_data = [tokenize_and_align_labels(words, labels) for words, labels in train_data]

# Para facilitar a criação do dataset, vamos "colapsar" a lista de dicionários em um único dicionário com listas
def collate_tokenized_data(tokenized_list):
    keys = tokenized_list[0].keys()
    collated = {key: [d[key] for d in tokenized_list] for key in keys}
    return collated

collated_train = collate_tokenized_data(tokenized_train_data)

# Criar um dataset customizado para NER
class NERDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.data.items()}
        return item
    def __len__(self):
        return len(self.data["input_ids"])

train_dataset = NERDataset(collated_train)

# --- Parte 3: Treinamento do modelo de NER ---

training_args = TrainingArguments(
    output_dir='./results_ner',
    num_train_epochs=3,
    per_device_train_batch_size=2,
    logging_dir='./logs_ner',
    evaluation_strategy="epoch",   # Avalia ao final de cada época
    logging_strategy="epoch",      # Registra os logs por época
    save_strategy="epoch",
    report_to="none"
)

data_collator = DataCollatorForTokenClassification(tokenizer)

# Callback para capturar métricas de cada época
from transformers import TrainerCallback

class EpochMetricsCallback(TrainerCallback):
    def __init__(self):
        self.epoch_metrics = []
    def on_evaluate(self, args, state, control, metrics, **kwargs):
        # Armazena as métricas com o número da época
        self.epoch_metrics.append({"epoch": state.epoch, **metrics})

epoch_callback = EpochMetricsCallback()

# Função para computar métricas de NER (no nível de token)
def compute_metrics_ner(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    true_labels = []
    true_preds = []
    for i in range(len(labels)):
        for j in range(len(labels[i])):
            if labels[i][j] != -100:
                true_labels.append(labels[i][j])
                true_preds.append(preds[i][j])
    acc = accuracy_score(true_labels, true_preds)
    prec = precision_score(true_labels, true_preds, average='weighted')
    rec = recall_score(true_labels, true_preds, average='weighted')
    f1 = f1_score(true_labels, true_preds, average='weighted')
    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=train_dataset,  # Neste exemplo, usamos os mesmos dados para avaliação
    data_collator=data_collator,
    compute_metrics=compute_metrics_ner
)

trainer.train()

model.save_pretrained("./model_saved")
tokenizer.save_pretrained("./model_saved")


def identify_genes_diseases(text, valid_genes):
    # Extrair tokens que podem conter letras, dígitos e hífens
    tokens = re.findall(r'\b[\w\-]+\b', text)
    genes_found = []
    for token in tokens:
        if sum(1 for c in token if c.isupper()) < 2:
            continue
        norm = normalize_gene(token)
        if norm in valid_genes:
            genes_found.append(norm)

    # Padrões para identificar doenças
    disease_patterns = {
        "COVID-19": r"\bCOVID-19\b",
        "Alzheimer's disease": r"\bAlzheimer(?:['’]s)? disease\b|\bAlzheimer\b",
    }
    diseases = []
    for disease, pattern in disease_patterns.items():
        if re.search(pattern, text, re.IGNORECASE):
            diseases.append(normalize_disease(disease))
    return genes_found, diseases

# Para cada abstract de treinamento, extrai os genes e doenças e imprime
all_genes = []
all_diseases = []
print("\n--- Extração dos genes e doenças dos textos de treinamento ---")
for i, text in enumerate(train_texts):
    genes, diseases = identify_genes_diseases(text, valid_genes)
    all_genes.extend(genes)
    all_diseases.extend(diseases)
    print(f"Texto de treino {i+1}:")
    print(f"  Genes identificados: {genes}")
    print(f"  Doenças identificadas: {diseases}")

# Plotar a frequência dos genes (acumulados de todos os abstracts)
gene_counts = Counter(all_genes)
plt.figure(figsize=(22, 6))
plt.bar(gene_counts.keys(), gene_counts.values(), color='lightgreen')
plt.xlabel('Genes Identificados')
plt.ylabel('Frequência')
plt.title('Frequência de Genes nos Textos de Treinamento')
plt.xticks(rotation=45)
plt.show()

# Plotar a frequência das doenças
disease_counts = Counter(all_diseases)
plt.figure(figsize=(8, 6))
plt.bar(disease_counts.keys(), disease_counts.values(), color='skyblue')
plt.xlabel('Doenças Identificadas')
plt.ylabel('Frequência')
plt.title('Frequência de Doenças nos Textos de Treinamento')
plt.xticks(rotation=45)
plt.show()

# -------------------------------------------------------------------
# Parte 5: Avaliação do desempenho do modelo (métricas) no dataset de teste

# Gerar dados de teste usando supervisão distante
test_texts = [
    "The ACE2 gene is linked to COVID-19 severity.",
    "LZTFL1 is associated with COVID-19",
    "APOE gene variants increase Alzheimer's risk.",
    "The PSEN1 gene increases the risk of having Alzheimer."
]
test_data = []

for text in test_texts:
    words, labels = get_bio_labels(text, valid_genes)
    test_data.append((words, labels))
    print("Teste - Texto:", text)
    print("Palavras:", words)
    print("Rótulos:", labels)
    print("-----")



# -------------------------------------------------------------------
# Parte 6: Extração de entidades em novos textos

# Função para identificar genes e doenças (baseada em regras) – já definida anteriormente:
def identify_genes_diseases(text, valid_genes):
    # Extrair tokens que podem conter letras, dígitos e hífens
    tokens = re.findall(r'\b[\w\-]+\b', text)
    genes_found = []
    for token in tokens:
        if sum(1 for c in token if c.isupper()) < 2:
            continue
        norm = normalize_gene(token)
        if norm in valid_genes:
            genes_found.append(norm)

    # Padrões para doenças: usar regex para detectar variações e normalizá-las
    disease_tokens = re.findall(r'\b[\w\'\-]+\b', text)
    diseases_found = []
    for token in disease_tokens:
        # Se o token corresponder a variações de COVID-19 ou SARS-COV-2, normaliza para "COVID-19"
        if re.fullmatch(r'(COVID[- ]?19|SARS[- ]?COV[- ]?2)', token, re.IGNORECASE):
            diseases_found.append("COVID-19")
        # Se o token corresponder a variações de Alzheimer (ex: "Alzheimer", "Alzheimer's", "Alzheimer disease")
        elif re.fullmatch(r'(Alzheimer(?:\'s)?(?:\s*disease)?)', token, re.IGNORECASE):
            diseases_found.append("ALZHEIMER’S DISEASE")
    return genes_found, diseases_found

# --- Para novos abstracts, use a função acima para extrair as entidades:

print("\n--- Extração de Entidades em Novos Abstracts ---")
new_test_texts = [
    "The ACE2 gene is linked to COVID-19 severity.",
    "LZTFL1 is associated with COVID-19",
    "APOE gene variants increase Alzheimer's risk.",
    "The PSEN1 gene increases the risk of developing Alzheimer."
]

for t in new_test_texts:
    genes, diseases = identify_genes_diseases(t, valid_genes)
    print(f"Abstract: {t}")
    print("  Genes identificados:", genes)
    print("  Doenças identificadas:", diseases)
    print("-----")
