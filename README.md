# BrainSegmentation

Project repository for tasks around brain segmentation.
The main goal of this project is to provide a consistent workflow for creating training datasets for neural network models.

The following workflow is an overview of the involved tasks:

```mermaid
---
config:
  layout: elk
  look: handDrawn
  theme: neutral
---

graph TD
    subgraph TRAINING_DATA_PIPELINE["Full Head Segmentation"]
        A["Input T1/T2 weighted image"]:::dataset --> B["CHARM skull/skin head segmentation"]:::segmentation
        A --> C["SynthSeg detailed brain segmentation"]:::segmentation
        B --> D["Label Merge"]:::analysis
        C --> D
        D --> E["Full Head Segmentation"]:::labelimage
    end

    T1["ADNI Dataset with</br>MPRAGE images"]:::dataset --> T2["Full Head Segmentation"]:::fullheadsegmentation
    T2 --> T3["Training Maps"]:::labelimage

    A1["Project MRI Scan"]:::dataset --> A2["Full Head Segmentation"]:::fullheadsegmentation
    A2 --> A3["Sufficient Segmentation"]:::labelimage
    A1 --> A4["Graylevel Distribution Analysis"]:::analysis
    A3 --> A4

    A4 --> G1["Brain Generator Configuration"]:::analysis
    G1 --> G2["Brain Generator"]:::analysis
    T3 --> G2

    G2 --> S1["Sythetic MRI Images"]:::output
    G2 --> S2["Ground Truth Labels"]:::output

    S1 --> M1["Neural Network Training"]:::analysis
    S2 --> M1


    %% Class Definitions
    classDef dataset fill:#FFD700,stroke:#B8860B,stroke-width:2px,color:#000,font-weight:bold;
    classDef fullheadsegmentation fill:#66CCFF,stroke:#007ACC,stroke-width:2px,color:#000,font-weight:bold;
    classDef segmentation fill:#aaeeFF,stroke:#007ACC,stroke-width:2px,color:#000,font-weight:bold;
    classDef labelimage fill:#99CC33,stroke:#558000,stroke-width:2px,color:#000,font-weight:bold;
    classDef analysis fill:#FF6666,stroke:#CC3333,stroke-width:2px,color:#000,font-weight:bold;
    classDef output fill:#FF9900,stroke:#CC6600,stroke-width:2px,color:#000,font-weight:bold;
    
    %% Subgraph Styling (Matching all :::fullheadsegmentation nodes)
    class TRAINING_DATA_PIPELINE fullheadsegmentation;
```
