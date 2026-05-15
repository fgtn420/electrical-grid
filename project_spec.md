Topic

The topic for your project is open to the any aspect of machine learning (i.e., some form of supervised and/or unsupervised learning is involved) which is connected to material taught in the class (this course). It is suggested that you choose or formulate a particular application/task/problem and move through the machine learning pipeline to solve it, but you may also study some aspect of machine learning from theoretical viewpoint.

Your project should revolve around a specific topic/question/problem statement. There should be some element of novelty/creativity (e.g., a new/modified dataset, a new question/new knowledge, a particular experimental setup, implementation, or interpretation and you must be unambiguous about your novelty vs what existed already) but be careful to find an appropriate scope; the grading criteria is mainly based on your ability to show that you understand machine learning, not to produce a research paper (which would additionally involve the criterion of potential impact to the scientific community -- this is not considered in the grading criteria for this project). 
Deliverables and Grading

Grading will be carried out on a presentation and the answers to questions and general discussion following it, to one of the senior teaching coordinators. Precise details will be announced closer to the end (it depends on how many teams there are), but it should normally be a short presentation + at least 10 minutes of questions and discussion.

Note: is your responsibility to form or find a team (but the teaching team is happy to help coordinate -- don't hesitate to ask us). Do not leave this task too late.  

What exactly will be graded? Grading is based on how well you convey your understanding of the machine learning aspects that relate to your project (and specifically those covered in the course). For example, if you created a new dataset and formulated a new problem, you'll talk a lot (and expect questions) about how you built/preprocessed/prepared it and used it to answer a particular question or overcome a particular problem. If you coded a decision tree algorithm from scratch, we'll discuss your implementation, it's running time complexity, performance, and many details about decision trees. If you compared the results of many algorithms from scikit-learn on benchmark datasets, then you will show good general knowledge about evaluation methods, different performance metrics, and the comparative advantages of the different methods and how this is reflected in the results. If you choose to tag on to an existing kaggle competition, you'll need to justify and motivate your chosen approach, and distinguish it from others approaches. Et cetera. 

Deliverables: What to submit here? A brief outline of your topic and team (see the template attached) as a single-page pdf. Plus (optionally) a version of your presentation slides in pdf form, additional experimental results, or any other material that didn't make it into the final presentation; all of which as a single pdf document. This submitted material is not graded directly but serves as a reference to the examiner during and after your presentation; to formulate questions (and to know what to ask about, and who best to direct the questions to), and to recall the main elements of your project later.
Suggestions and Hints

A rough outline of how to proceed (only as a suggestion):

    Define the question/problem statement you will approach
    Obtain/curate a suitable dataset (ensure that you have permission to gather this data); check that is sufficient to answer 1.
    Choose/create: performance metric(s)/loss function(s), model(s), algorithm(s) you will use/build
    Write the code and set up the experimental/analytical framework
    Run experiments to answer the question/evaluate success (negative answers are fine if well-explained), illustrate, and carefully interpret the results.
    Additionally you could pull out some 'nuggets of knowledge' from your analysis (findings that were perhaps unexpected/surprising, particularly interesting, or provocative; something you learned about the problem)
    Identify limitations, speculate about future steps, and what you have learned/would have done differently
    Revisit all the above steps again to fine-tune everything.

Hints:

    Go for a topic that you are interested in or want to learn more about
    Prompting LLMs is, in itself, not machine learning, we prefer that you train a rudimentary simplistic language model, rather than simply trying to fit an  LLM into your pipeline 
    Try to avoid having to work with large volumes of non-tabular data (e.g., images) on the large architectures which imply its use; or at least be aware of the challenges: it requires substantial computation, tiresome hyper-parameter tuning; and interpretation of results can be challenging.
    Do not do a project on reinforcement learning (which comes at the end of the course).
    Make sure to limit the scope appropriately. Top reason for top teams not getting top grades was being too ambitious, trying to do too much and not conclusively achieving (or not concisely defining) any of their original objectives. 
    In experimental comparisons, never forget to compare to some kind of baseline (linear regression, Naive Bayes, predicting the majority class, etc.) as a reference
    Focus on interpretation of results rather than simply reporting them
    Recognize limitations of your study (indeed, make sure to limit it to a suitable scope -- given the limited time frame)
    Complex questions and discussion arise from relatively simple methods and tasks -- yet another reason to start simple, and aim to set a clear limited scope
    It is often the data which makes an interesting problem. Data is everywhere: you may gather it 'manually' (data-entry from observations, questionnaires, ...) or get it from some online source, or some offline source, data repositories, ... You may turn an existing collection into a dataset, modify an existing dataset, or just take an off-the-shelf (e.g., from Kaggle) dataset.
    If you do collect/curate your own data (and this is great!), make sure it is enough (at least several hundred data points -- but this depends greatly on the problem/context) and make sure it is really suited to your objectives

If you're not sure about anything, don't hesitate to discuss with one of the teaching coordinators prior to committing to a project, or a particular idea/approach.

It is of utmost importance to unambiguously mention/reference/cite whatever you based your project on (blog post, code on github, some existing Kaggle competition, public data set, ...). You can copy figures, copy code, take data, from anywhere you like as long this is permitted (e.g., the data/algorithm is not legally restricted), as long as you explicitly acknowledge this clearly (in your slides/presentation).  