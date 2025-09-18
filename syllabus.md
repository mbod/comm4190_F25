

# COMM4190/5190 "Talking with AI" - Computational and Communication Approaches


## Fall 2025


### Mon/Wed 5.30-7.00pm (Room: ASC 109)


### Professor: Matt O'Donnell

-   Email: mbod@asc.upenn.edu
-   Office Hours: via Zoom (see link in Canvas)
    -   Specific times TBA
        -   But this may vary and I will always try and be available for
            appointments outside these times if you email with sufficient
            notice.


### TA: Cameron Moy

-   Email: cameron.moy@asc.upenn.edu
-   Office Hours:
    -   Thursday 1-3pm in ASC 136 and on Zoom (see link in Canvas)

---


# Course Description Goals and Objectives

Increasingly, our daily communications involve responding to and
interacting with language produced by artificial intelligence
models. On the surface, large language models (LLMs) and generative AI
(genAI) tools (e.g ChatGPT, Claude, Llama, Gemini, etc.) appear to
have crossed a milestone in terms of their human-like ability to
generate coherent and idiomatic texts. This has significant
implications (both positive and negative) for human communication
systems and their products, from creative fiction to news, from
academic texts to social media content. It also raises many questions
around whether we can identify, trust, learn from and use AI generated
language. We find ourselves "talking with AI" in at least two senses
of the phrase: 1. Using genAI tools to help us communicate (better?)
(i.e. genAI as collaborator/co-pilot) and 2. Encountering these tools
as communicative partners (i.e. situations in which human-human
interactions are increasingly becoming human-AI interactions).

In this course, we will begin to answer these questions in two
ways:

1.  Analyzing Key Issues: Drawing upon relevant frameworks in
    communication and language theory, we will explore the
    transformative nature of AI-generated communication and its impact
    on individuals and society.
2.  Hands-on Application: In parallel, students will acquire skills
    using Python to interact with machine and deep learning models
    (particularly LLMs) to better understand how they work and explore
    their abilities and limitations. We will work with code to work
    with and finetune various AI models and consider common
    applications, such as a simple voice assistant, image classifier,
    misinformation identifier, and a basic text generative application.

Through this course students will be equipped for a range of contexts
impacted by developments in AI. The course expects students to have
basic experience in Python coding and using Jupyter notebooks.


## Objectives

Through this course students will:

-   Develop an understanding of how LLMs work at various conceptual
    levels (above the low-level technical/mathematical), including the
    three stages of training (pre-training, fine-tuning and RLHF) and
    the key component of these models, namely the `Transformer`
    (self-attention).

-   Consider questions of:
    -   `how` and `what` LLMs learn during training and ongoing usage
    -   what level and kind of (if any) understanding and representation
        of meaning do LLMs exhibit and actually have
    -   to what extent are LLMs able to reason, plan, create models of the
        world and others (e.g. Theory of Mind)
    -   how LLMs behave in various communicative contexts in relation to
        relevant Communication theories (e.g. interpersonal and
        social/cultural theories etc)
    -   what kinds of `emergent` knowledge and behaviors LLMs seem to
        exhibit as they increase in size (of both data and parameters) and
        the implications of these phenomena

-   Learn the basic theoretical framework and concepts from pragmatics
    (the theory of `meaning in context` in language use) and consider
    and test `if` and `how` LLMs behave in accordance to these


# Assessment


## Overview of assignments

****NOTE**** - IF YOU ARE TAKING THIS CLASS AS COMM5190 (Graduate Level)
there may be a slightly different balance of assignments allowing you
to develop a topic of interest in more detail.

-   Prompt/response and reading blogging (30%)

-   "Living with a book" reading assignment (10%)

-   Coding and task assignments (25%)

-   Final Project (35%)


## Details

1.  Prompt/response and readings blog (30%)
    -   Over the course of the semester you will create 2 or 3 blog posts
        a week using the notebook blogging system setup to generate a
        github pages blog
    -   Each blog post should be a short and engaging analysis exploring topics such as:
        -   how LLMs respond to specific prompts
        -   whether they are are able or unable to mirror certain human
            communicative behaviors (e.g. recognize and use idioms)
        -   demonstrate the use of an LLM for a specific task
            (e.g. generating product names, descriptions and branding)
    -   ****FOR COMM5190 students****
        -   The blog will also be used for reading responses to the discussion articles
        -   For each of the discussion articles (see below) you should
            write a response post summarizing your thoughts on the
            article and the broader issues in genAI and communication.
    -   Students are encouraged to follow each others blogging
        -   Example blogs from previous class:
            -   <https://ekeogh03.github.io/comm4190_S24_Using_LLMs_Blog/>
            -   <https://yummyhopper.github.io/comm4190_S24_Using_LLMs_Blog/>
            -   <https://emmaluo3.github.io/comm4190_S24_Using_LLMs_Blog/>

2.  "Living with a book" Assignment (10%)
    -   The goal of this assignment is have you spend time reading **and
        writing in** (e.g. underlining, making notes and drawing etc) in a
        physical book that you will carry around with you and bring to every class.
    -   So you **MUST BUY A PHYSICAL COPY OF YOUR SELECTED BOOK**.
    -   You will be assigned to read one of the following books across this semester.
        -   The goal is to have an equal number of students reading each of the books.
        -   You will have a opportunity to rank your preferences and we will try to make sure you are assigned to one of your top preferences.
        -   You shouldn't have read the book before.
    
    -   **Book selection**
        -   Buolamwini, Joy (2024). *Unmasking AI: My Mission to Protect What Is Human in a World of Machines*. Random House.
        -   Mollick, Ethan (2024). *Co-Intelligence Living and Working with AI*. Penguin.
        -   Summerfield, Christopher (2025). *These Strange New Minds. How AI Learned to Talk and What It Means*. Penguin.
        -   Bender, Emily & Hanna, Alex (2025). *The AI Con. How to Fight Big Tech's Hype and Create the Future We Want*. HarperCollins.
        -   Bennett, Max (2023). *A Brief History of Intelligence Evolution, AI, and the Five Breakthroughs That Made Our Brains*. HarperCollins.
        -   Hoffman, Reid & Beato, Greg (2025). *Superagency. What Could Possibly Go Right with Our AI Future*. Authors Equity.
        -   Ananthaswamy, Anil (2024). *Why Machines Learn. The Elegant Math Behind Modern AI*. Penguin.
    
    -   More details on how this assignment will work will be discussed in class.

3.  Coding and Task Assignments (25%)
    -   There will be a small number of coding assignments to help
        practice the use of LLMs through APIs, do batch processing of
        prompts, evaluate responses and models, creating a RAG LLM system
        and simple finetuning
    -   In addition, there will some assignments focused on using LLMs to
        generate particular content (e.g. persuasive health messages) or
        emulate common communicative tasks.
    -   Assignments will either be released as Jupyter notebooks and
        submitted through JupyterHub OR as Canvas assignments
    -   We will use github repositories for content to help you build up
        a portfolio of example work.
        -   Some examples from previous class:
            -   <https://github.com/jason-saito/Health-Messages-Project>
            -   <https://github.com/smliu-hub/healthcomm_campaign>
            -   <https://github.com/SeanMcKeown1/healthcomm_BUGS>-
            -   <https://github.com/kendallen/comm_4190_S24_expert_presentations>
            -   <https://github.com/Lbrienza4498/Expert_Presentations>

4.  Final Project (35%)
    -   This is a group project. You can build your own team (of two or
        three) or be assigned to a group based on interest.
    -   The goal of this project is for you to create an LLM-based
        application, focused on a particular task or communicative
        context.
    -   Example projects from previous class:
        -   <https://github.com/eakadiri/COMM4190-Final-Project---Developing-a-Triage-ChatBot>-
        -   <https://github.com/ekeogh03/CollegiateCapital_project>
        -   <https://github.com/Lbrienza4498/final_4190>


# Textbooks and recommended readings


## Textbooks

Alammar, J. & Grootendorst, M. (2024) *Hands-On Large Language Models*.

-   This book is references as `HOLLM` in the schedule below.
-   It is available online through the library and the UPenn O'Reilly subscription

Christiansen, M.H. and Chater, N. (2022) *The Language Game: How
Improvisation Created Language and Changed the World*. Basic Books.

-   This book is references as `LGAME` in the schedule below.

Mitchell, M. (2019) *Artificial Intelligence: A Guide for Thinking Humans*.

-   This book is referenced as `AITH` in the schedule below.

Phoenix, J. & Taylor, M.(2024) *Prompt Engineering for Generative AI*. O'Reilly Media, Inc.

-   This book is referenced as `PEGAI` in the schedule below.
-   It is available online through the library and the UPenn O'Reilly subscription

Kaplan, J. (2024) *Generative Artificial Intelligence: What Everyone Needs to Know*. Oxford University Press

-   This book is available online through the library:
    <https://whateveryoneneedstoknow.com/display/10.1093/wentk/9780197773536.001.0001/wentk-9780197773536>
-   Reference to it in the schedule is as `GAI`

Togelius, J. (2024) *Artificial General Intelligence*. MIT Press Essential Knowledge Series

-   Reference to this book in the schedule is as `AGI`


## Other readings

Enfield, N. (2017). *How we talk: The inner workings of
conversation*. Basic Books.

Wolfram, S. (2023) *What Is ChatGPT Doing … and Why Does It
Work?*. Blog post February 14, 2023

-   <https://writings.stephenwolfram.com/2023/02/what-is-chatgpt-doing-and-why-does-it-work/>


## Discussion Readings (For COMM5190 Graduate level)

-   If you are taking this class at graduate level as COMM5190, you will
    be required to read and engage in discussion of these
    articles. Students will be assigned at least one for which they will
    be responsible to provide a brief summary and lead discussion.

-   You will also write a short reading response using your blog (see above).

-   A set of recent and relevant articles will be selected based on key
    topics covered and student interests (Please let instructor know of
    specific articles you might be interested in the class reading). The
    articles listed below are some of the key articles that could be
    used. But the field is developing rapidly so new ones are likely to
    be added.


### Key articles

-   Bender, Emily M., Timnit Gebru, Angelina McMillan-Major, and
    Shmargaret Shmitchell. “On the Dangers of Stochastic Parrots: Can
    Language Models Be Too Big? 🦜.” In Proceedings of the 2021 ACM
    Conference on Fairness, Accountability, and Transparency,
    610–23. FAccT ’21. New York, NY, USA: Association for Computing
    Machinery, 2021. <https://doi.org/10.1145/3442188.3445922>.

-   Chang, Tyler A., and Benjamin K. Bergen. “Language Model Behavior: A
    Comprehensive Survey.” arXiv, August
    25, 2023. <https://doi.org/10.48550/arXiv.2303.11504>.

-   Kim, Junghwan, Jinhyung Lee, Kee Moon Jang, and Ismini
    Lourentzou. “Exploring the Limitations in How ChatGPT Introduces
    Environmental Justice Issues in the United States: A Case Study of
    3,108 Counties.” Telematics and Informatics 86 (February 1,
    2024): 102085. <https://doi.org/10.1016/j.tele.2023.102085>.

-   Mitchell, Melanie, and David C. Krakauer. “The Debate Over
    Understanding in AI’s Large Language Models.” Proceedings of the
    National Academy of Sciences 120, no. 13 (March 28, 2023):
    e2215907120. <https://doi.org/10.1073/pnas.2215907120>.

-   Schaeffer, Rylan, Brando Miranda, and Sanmi Koyejo. “Are Emergent
    Abilities of Large Language Models a Mirage?” arXiv.org, April
    28, 2023. <https://arxiv.org/abs/2304.15004v2>.

-   Song, Yuanfeng, Yuanqin He, Xuefang Zhao, Hanlin Gu, Di Jiang,
    Haijun Yang, Lixin Fan, and Qiang Yang. “A Communication Theory
    Perspective on Prompting Engineering Methods for Large Language
    Models.” arXiv, October
    23, 2023. <https://doi.org/10.48550/arXiv.2310.18358>.

-   Soni, Nikita, H. Andrew Schwartz, João Sedoc, and Niranjan
    Balasubramanian. “Large Human Language Models: A Need and the
    Challenges.” arXiv, May 9, 2024. <http://arxiv.org/abs/2312.07751>.

-   Törnberg, Petter. “How to Use LLMs for Text Analysis.” arXiv, July
    24, 2023. <https://doi.org/10.48550/arXiv.2307.13106>.

-   Trott, Sean, and Cameron Jones. “Do Large Language Models Have a
    ‘Theory of Mind’?” Substack newsletter. The Counterfactual (blog),
    September
    14, 2023. <https://seantrott.substack.com/p/do-large-language-models-have-a-theory>.

-   Vaswani, Ashish, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion
    Jones, Aidan N. Gomez, Lukasz Kaiser, and Illia
    Polosukhin. “Attention Is All You Need.” arXiv, August
    1, 2023. <https://doi.org/10.48550/arXiv.1706.03762>.

-   Wei, Jason, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Brian
    Ichter, Fei Xia, Ed H Chi, Quoc V Le, and Denny
    Zhou. “Chain-of-Thought Prompting Elicits Reasoning in Large
    Language Models,”

-   Yiu, Eunice, Eliza Kosoy, and Alison Gopnik. “Imitation versus
    Innovation: What Children Can Do That Large Language and
    Language-and-Vision Models Cannot (Yet)?” arXiv, May
    8, 2023. <https://doi.org/10.48550/arXiv.2305.07666>.

-   Yu, Zihan, Liang He, Zhen Wu, Xinyu Dai, and Jiajun Chen. “Towards
    Better Chain-of-Thought Prompting Strategies: A Survey.” arXiv,
    October 7, 2023. <https://doi.org/10.48550/arXiv.2310.04959>.


# Schedule


## **NOTE** This is a tentative schedule is will be updated as the course progresses


## Class structure

-   Usually each class session will be divided into two parts:

-   **Content and Discussion** - This part of the class will introduce
    and review the key topics outlined in the schedule. It is important
    that you read the assigned material BEFORE the class as you'll be
    expected to engage in discussion and other activities based on this
    content.
-   **Lab activities** - The second part will be focused on practical
    exercises to put into practice what we have been learning about
    thinking about, collecting, analyzing, interpreting and
    communicating data. This will include learning some basic steps
    using Python and R scripts and some other tools for data analysis
    and visualization.


## Week 1 - Introduction and Setup


### Wednesday 27 August

-   Introduction to and overview of the class
-   Setup and testing of JupyterHub
    -   ****IMPORTANT**** Make sure you have completed initial survey quiz in Canvas to get
        your github userid setup on the class server.


## Week 2 - Introduction and Setup

READINGS:

-   `LGAME` Ch. 1 pp. 7-19


### Monday 20 January ****NO CLASS Labor Day****

**IMPORTANT** - PLEASE MAKE SURE TO COMPLETE BEFORE CLASS 

-   Follow the 4 steps in notebooks in the
    `03_SETUP_Blog_and_github_access` folder in JupyterHub to
    1.  setup SSH key for github access from `commjhub.asc.upenn.edu`
    2.  fork and clone the blog repo from github
    3.  test the blogging steps


### Wednesday 03 September

-   Language as interaction and exchange
    -   Turn types in conversation
        -   Statements & Questions
        -   Offers & Demands
    -   Responses: Compliance vs Challenge

-   **Exercises**
    -   Record and analyze conversation
    -   Using `openai` API from Python
    -   Python Basics using `turtle` module


## Week 3 - Prompting and How LLMs work (Part 1)

READINGS:

-   `PEGAI` Ch. 1

(<https://learning.oreilly.com/library/view/prompt-engineering-for/9781098153427/ch01.html>)

-   Wolfram 2023 Blog post *What is ChatGPT Doing*
-   Chapter 1 of *The Language Game* - Christiansen & Chater (Ch. 1)
-   `GAI` Ch. 2


### Monday 08 September

-   Understanding what LLMs are doing - next word prediction
-   Some challenges for LLMs
    -   spatial logic
    -   theory of mind (ToM) scenarios

-   Looking at GPT2
    -   Word embeddings

-   How LLMs work (Part 1)
    -   High level view  of neural networks and deep learning
    -   Input: Representation
    -   Model (neural network): Transformations
    -   Output: Probabilistic distribution over finite set of values (e.g. categories, vocabulary)
-   Read: Wolfram article - Python examples using GPT2
-   Word Embeddings
-   Language as contextual negotiation
    -   *The Language Game* - learning and using language as charades


### Wednesday 10 September

-   Getting setup for blogging assignments
    1.  setup SSH key for github.com on commjhub server
    2.  fork blog repository
    3.  understand and practice creating post and pushing to github and viewing on github pages


## Week 4 - Prompting and How LLMs work (Part 2)

READINGS:

-   `PEGAI` Ch. 2 (<https://learning.oreilly.com/library/view/prompt-engineering-for/9781098153427/ch02.html>)

-   Watch this video: <https://www.youtube.com/watch?v=OFS90-FX6pg>


### Monday 15 September

-   Overview of Prompt Engineering
-   Overview of how LLMs are built
    1.  Pretraining
        -   Self-supervised
        -   Next word prediction
    2.  Finetuning
        -   Domain
        -   Task
        -   Instruction
    3.  Reinforcement Learning
        -   RLHF: Reinforcement Learning from Human Feedback
-   How LLMs work (continued)
    -   Embeddings

-   Exercise: Using LLMs to create a branding for a product


### Wednesday 17 September

-   How do LLMs work
    -   Dealing with context
    -   Transformer and self-attention
-   How LLMs are trained
    -   Stage 1: Pretraining

-   Open vs Closed (Proprietary) models
-   Tool: <https://chat.lmsys.org>
    -   Comparing responses from multiple LLMs

-   Running local LLMs
    -   Ollama (ollama.ai)


## Week 5 - How LLMs are trained (Part 1) and does conversation with an LLM work?

READINGS: 

-   `PEGAI` Ch. 3 (<https://learning.oreilly.com/library/view/prompt-engineepring-for/9781098153427/ch03.html>)

-   Enfield (Ch. 1)


### Monday 22 September

-   Overview of LLM model landscape
-   Using <span class="underline">Hugging Face</span>
    -   Register for an account on <https://huggingface.co/>


### Wednesday 24 September

-   How does human-human conversation work?
-   Is a ChatBot really engaging in conversation?
-   Finetuning with Question-Answer data


## Week 6 - How LLMs are trained - Finetuning (Part 2)

READINGS:

-   Christiansen & Chater (Ch. 2)
-   Enfield (Ch. 2)


### Monday 29 September

-   How LLMs are trained
    -   Stage 2: Fine Tuning
        1.  Domain
        2.  Task
        3.  Instruction
-   From next word prediction tool to chatbot


### Wednesday 01 October

-   Comparing a base model with a fine tuned model
    -   Examples:
        -   Gemma
        -   Llama

-   TOOL: Using Ollama to run local models


## Week 7 - LLMs and image models

READINGS:

-   `PEGAI` Ch. 7 (<https://learning.oreilly.com/library/view/prompt-engineepring-for/9781098153427/ch07.html>)


### Monday 06 October

-   How do image-to-text and text-to-image models work?
-   Diffusion vs Transformer models


### Wednesday 08 October

-   Using DALL-E3/image-gpt-1, Imagen-3 and MidJourney


## ****FALL BREAK (October 9-12)****


## Week 8 - Using LLMs for persuasive communication

READINGS:

-   `PEGAI` Ch. 8 (<https://learning.oreilly.com/library/view/prompt-engineepring-for/9781098153427/ch08.html>)


### Monday 13 October

-   Examples of persuasive communication for Public Health
-   Prompting strategies for generating persuasive messages


### Wednesday 15 October


## Week 9 - Arguing with LLMs & Langchain (Part 1) and Gradio for building LLM applications

READINGS:

-   `PEGAI` Ch. 4 (<https://learning.oreilly.com/library/view/prompt-engineepring-for/9781098153427/ch04.html>)


### Monday 20 October (**\* ROOM CHANGE: IN ASC 108 \***)

-   Choice dilemma task
-   Group consensus


### Wednesday 22 October

-   Using \`langchain\` and \`langgraph\`


## Week 10 - AGI, LLMs and Knowledge Seeking (cont)

READINGS: 

-   `AGI` Chapters 1-3


### Monday 27 October

-   Using RAG in \`langchain\`
-   Web-based RAG


### Wednesday 29 October

-   Integrating search and LLMs
    -   SearchGPT, Gemini, MS Co-pilot, meta.ai
    -   Perplexity


## Week 11 - LLMs and Knowledge Seeking

READINGS: 
  = `AGI` Chapter 4-5


### Monday 03 November

-   Retrieval Augmented Generation (RAG)
-   RAG using \`langchain\`


### Wednesday 05 November

-   Retrieval Augmented Generation (RAG)
-   RAG using \`langchain\`
-   Embeddings and vector databases
-   Large context prompts  vs chunking and RAG


## Week 12 - LLMs as agents (Part 1)

READINGS:

-   `PEGAI` Ch. 6 (<https://learning.oreilly.com/library/view/prompt-engineepring-for/9781098153427/ch06.html>)


### Monday 10 November

-   What are AI agents
-   Examples of tasks using agents


### Wednesday 12 November (**\* ROOM CHANGE: IN ASC 108 \***)

-   LangChain, LangGraph, crewAI


## Week 13 - LLMs as agents (Part 2) and genAI applications

READINGS: 

-   `GAI` Chapters 3&4
-   `PEGAI` Ch. 10 (<https://learning.oreilly.com/library/view/prompt-engineepring-for/9781098153427/ch10.html>)


### Monday 17 November


### Wednesday 19 November


## Week 14 - Impacts and Ethics of genAI

READINGS:

-   `GAI` Chapters 5&6


### Monday 24 November

-   Bias in pretraining
-   Bias in finetuning
-   Alignment/censoring


### Wednesday 26 November

-   Using uncensored models
-   Making LLMs 'forget'


### ****THANKSGIVING BREAK (November 27-30)****


## Week 15 - Summary and Project Presentations


### Monday 01 December

-   Group project presentations
-   Future prospects for LLMs


### Wednesday 03 December

-   Group project presentations


### ASSIGNMENTS DUE - TBA

