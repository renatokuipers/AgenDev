flowchart TD
  %% ----------------- PERCEPTION MODULE -----------------
  subgraph Perception_Module["Perception Module"]
    SP["Sensory Processing"]
    II["Interpretation & Integration"]
    SP -->|Provides raw sensory data| II
  end

  %% ----------------- CONSCIOUSNESS MODULE -----------------
  subgraph Consciousness_Module["Consciousness Module"]
    AA["Awareness & Attention"]
    FDA["Focused vs. Divided Attention"]
    AA -->|Refines attention| FDA
  end

  %% ----------------- UNCONSCIOUS/ PRECONSCIOUS MODULE -----------------
  subgraph Unconscious_Module["Unconscious/Preconscious Processes"]
    AP["Automatic Processing"]
    IM["Implicit Memory"]
  end

  %% ----------------- MEMORY SYSTEMS MODULE -----------------
  subgraph Memory_Systems_Module["Memory Systems"]
    WM["Working Memory"]
    LTM["Long-Term Memory<br/>(Episodic, Semantic, Procedural)"]
    MC["Memory Consolidation"]
    WM -->|Temporary storage for immediate info| LTM
    LTM -->|Updated via consolidation| MC
  end

  %% ----------------- LANGUAGE PROCESSING MODULE -----------------
  subgraph Language_Module["Language Processing"]
    LC["Language Comprehension"]
    LP["Language Production"]
    LC -->|Feedback loop| LP
    LP -->|Reinforces understanding| LC
  end

  %% ----------------- COGNITIVE PROCESSES MODULE -----------------
  subgraph Cognitive_Module["Cognitive Processes"]
    PR["Problem Solving & Reasoning"]
    LE["Learning"]
    AC["Attention Control"]
    PR -->|Drives adaptive learning| LE
    LE -->|Updates focus strategies| AC
  end

  %% ----------------- EMOTIONAL PROCESSES MODULE -----------------
  subgraph Emotional_Module["Emotional Processes"]
    EG["Emotion Generation"]
    ER["Emotion Regulation"]
    VA["Valence & Arousal Evaluation"]
    EG -->|Initiates affective responses| ER
    ER -->|Modulates emotional intensity| VA
  end

  %% ----------------- MOTIVATION & DRIVES MODULE -----------------
  subgraph Motivation_Module["Motivation and Drives"]
    IP["Incentive Processing"]
    GS["Goal Setting"]
    IP -->|Influences drive formation| GS
  end

  %% ----------------- EXECUTIVE FUNCTIONS MODULE -----------------
  subgraph Executive_Module["Executive Functions"]
    PD["Planning & Decision-Making"]
    IC["Inhibitory Control & Cognitive Flexibility"]
    TS["Task Switching"]
    PD -->|Directs control| IC
    IC -->|Enables adaptive switching| TS
  end

  %% ----------------- METACOGNITION MODULE -----------------
  subgraph Metacognition_Module["Metacognition"]
    SM["Self-Monitoring"]
    RT["Reflective Thinking"]
    SM -->|Evaluates performance| RT
    RT -->|Adapts strategies| SM
  end

  %% ----------------- PERSONALITY & IDENTITY MODULE -----------------
  subgraph Personality_Module["Personality and Identity"]
    TT["Traits and Temperaments"]
    SC["Self-Concept & Narrative Identity"]
    TT -->|Shapes character| SC
    SC -->|Reinforces trait expression| TT
  end

  %% ----------------- SOCIAL COGNITION MODULE -----------------
  subgraph Social_Module["Social Cognition"]
    TM["Theory of Mind"]
    EP["Empathy & Perspective Taking"]
    SI["Social Interaction Models"]
    TM -->|Informs social insight| EP
    EP -->|Drives interaction models| SI
  end

  %% ----------------- DECISION-MAKING PROCESSES MODULE -----------------
  subgraph Decision_Module["Decision-Making Processes"]
    RR["Risk and Reward Assessment"]
    HD["Heuristics vs. Deliberative Thought"]
    MR["Moral Reasoning"]
    RR -->|Feeds into quick heuristics| HD
    HD -->|Balances ethical considerations| MR
  end

  %% ----------------- CREATIVITY & IMAGINATION MODULE -----------------
  subgraph Creativity_Module["Creativity and Imagination"]
    DT["Divergent Thinking"]
    SH["Simulation of Hypotheticals"]
    DT -->|Inspires scenario simulation| SH
    SH -->|Enhances creative problem solving| DT
  end

  %% ----------------- ERROR MONITORING MODULE -----------------
  subgraph Error_Module["Error Monitoring & Adaptive Control"]
    CD["Conflict Detection"]
    EC["Error Correction"]
    CD -->|Identifies discrepancies| EC
    EC -->|Feeds corrections back| CD
  end

  %% ----------------- TEMPORAL PROCESSING MODULE -----------------
  subgraph Temporal_Module["Temporal Processing"]
    TP["Time Perception"]
    SQ["Sequencing"]
    TP -->|Organizes events| SQ
  end

  %% ================== INTER-MODULE CONNECTIONS ==================

  %% Perception -> Consciousness
  II -->|Sends processed input| AA
  SP -->|Direct sensory input| AA

  %% Consciousness distributes to Memory & Language
  AA -->|Selects focus for storage| WM
  AA -->|Supplies context| LC

  %% Perception informs Unconscious Processes
  II -->|Implicit cues| AP
  II -->|Subtle patterns| IM

  %% Focused attention (FDA) enhances Working Memory
  FDA -->|Optimizes info storage| WM

  %% Language interconnects with Cognitive Processes
  LC -->|Provides linguistic context| PR
  LP -->|Contributes to reasoning| PR

  %% Memory Systems support Cognitive Processes
  WM -->|Immediate info for reasoning| PR
  LTM -->|Background knowledge| PR

  %% Memory Consolidation enhances Learning
  MC -->|Integrates new experiences| LE

  %% Cognitive Processes influence Emotional Processes
  PR -->|Triggers emotional responses| EG
  LE -->|Modulates emotional intensity| EG
  AC -->|Regulates cognitive focus affecting emotions| EG

  %% Memory and Emotion Interaction
  WM -->|Carries emotional context| VA
  LTM -->|Informs emotional appraisal| VA

  %% Emotional Processes drive Motivation
  EG -->|Activates drive signals| IP
  ER -->|Modulates drive intensity| IP
  VA -->|Shapes incentive values| IP

  %% Motivation influences Executive Functions
  IP -->|Drives planning| PD
  GS -->|Sets targets for planning| PD
  IP -->|Feedback into goal refinement| GS

  %% Executive Functions are monitored by Metacognition
  PD -->|Monitored by| SM
  IC -->|Assessed by| SM
  TS -->|Evaluated by| RT
  SM -->|Provides feedback to| PD
  RT -->|Refines planning strategies| PD

  %% Cognitive Processes and Metacognition interact
  PR -->|Monitored for efficiency| SM
  LE -->|Reflected upon in| RT
  RT -->|Adjusts attention control| AC

  %% Error Monitoring feeds back into Cognitive & Executive Modules
  CD -->|Detects errors in| PR
  CD -->|Alerts| SM
  EC -->|Corrects reasoning paths in| PR
  EC -->|Updates planning in| PD

  %% Decision-Making Processes formation
  PD -->|Evaluates outcomes via| RR
  PD -->|Considers intuitive vs. analytic paths via| HD
  PD -->|Assesses moral implications via| MR
  CD -->|Impacts risk evaluation| RR
  EC -->|Shapes heuristic choices| HD

  %% Decision-Making interacts with Social Cognition & Personality
  RR -->|Informs social expectations| TM
  RR -->|Influences empathic responses| EP
  HD -->|Relates to interaction models| SI
  MR -->|Shapes self-concept| SC
  MR -->|Modulates trait expression| TT

  %% Social Cognition and Personality interplay
  TM -->|Contributes to personal identity| SC
  EP -->|Enhances narrative identity| SC
  SI -->|Reflects underlying temperaments| TT

  %% Social modules support Creativity
  TM -->|Inspires divergent thinking| DT
  EP -->|Drives creative ideation| DT
  SI -->|Supports hypothetical simulation| SH
  SC -->|Feeds into creative processes| DT
  TT -->|Encourages novel approaches| DT
  SC -->|Informs simulation scenarios| SH

  %% Creativity enriches Cognitive Processes
  DT -->|Feeds back into reasoning| PR
  SH -->|Boosts adaptive learning| LE

  %% Temporal Processing organizes overall flow
  TP -->|Structures sequence of events| SQ
  SQ -->|Guides planning sequences| PD
  SQ -->|Orders memory storage in| WM
  SQ -->|Enhances consolidation process| MC

  %% Cross-Module Feedback Loops
  EG -->|Triggers further automatic responses in| AP
  VA -->|Modulates implicit memory in| IM
  LC -->|Supplies language context for reflection| RT
  LP -->|Informs narrative identity in| SC
  SM -->|Feeds back to adjust language processing| LC
  RT -->|Adapts attention control mechanisms in| AC
  AC -->|Regulates input to working memory| WM