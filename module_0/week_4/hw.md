1. According to the author (me) should you treat the number of neurons and layers as a searchable hyperparameter? Why or why not? Do you agree (it's okay to disagree, just have a good argument!)?
* I agree. One is that if you have too few neurons that you will not be able to take into account non-linearity in the data. When the nerual network is too simple then the activation functions will not be recieve enough data to learn in backprop. The other side of the coin, too many neurons can saturate the neurons causing no further learning on new data. On the choice of deep vs wide network I really have no idea. Guessing, I can see a deep network important for some use cases where all the information is not important, maybe feature dependency is low, and the loss of some information as weights are narrowed is important. On the other had a wide network I just don't know.

2. In the lecture Andrew Ng describes two separate processes you could use for hyperparamemter search.  He describes these as a 'panda' method and a 'caviar' method.  What are these two methods?
* Panda - baby sit one model, train one model over several weeks, typically when you do not have a lot of resources 
* Cavier - training many models in parallel, have a lot of resources vs amount of data, run different models at the same time with different hyperparameter settings to pick the best one 


3. Compare and contrast scikit-learn's gridsearch and randomsearch.  What are the pros and cons of each?
* randomsearch - not all parameters are tried out, faster, specify the number of hyperparametrs and it will select a subset, much faster than gridsearch, but still not inexpensive, con is this is entirely random could produce the non-optima solution and different results each pass
* gridsearch - brute force, define a set of hyperparametrs and train the model for all combos selecting the best one, good for a fast model very bad for a long running network

4. How does "hyperband" work?
* Performs random sampling. First and/or second iteration hyperband runs config, taking the best performers and runs those longer
* the ability to sample a hyperparam config, ability to train a particular hyperparam config until it has reached a given number of iterations
* https://people.eecs.berkeley.edu/~kjamieson/hyperband.html

5.  On your own, check out Bayesian optimization for hyperparameters and genetic algorithms for hyperparameter optimization.  Do either of these methods seem promising for optimizing faster than brute force?
* In a sense yes I think either of the hyperparameters would be better than brute force. I would say it depends. Ignoring best and worst case scenerios, bayesion optimization offers a more resource intensive optimization with with the exception of early stopping. If appropriate bounds were known the use probability of prior runs to determine optima convergence makes sense and sounds great. Testing on a data would have to prove this out. On the other hand, genetic algorithms, GA, for hyperparameter optimization is extremely interesting. The process of selection, mimicking darwiniasm, by removing poorly performing hyperpameters and taking the best performing hyperparameters to use in the next run (reproduction) makes a lot of sense, crossover to mix some of those best hyperparameters, and mutation adding some variation to create generality. Although purely based on this research, http://www.flll.jku.at/div/teaching/Ga/GA-Notes.pdf, it appears GAs are only optima in problems where the optimization is non-conventials and the use of derivatives would not be possible. Without having enough time to test and see performance I rely on published papers which show promise for both methods, with the catch that 

