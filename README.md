### Antonio Feltrin (2097126), Simone Toso (2095484)


# A bit of theory

Our aim is to *classify* some phrases, using a predetermined set of categories $\mathcal{C}$ (e.g. `{reliable, unreliable}`).

To do so, we must learn a *classification function* $\gamma: \mathcal{X} \to \mathcal{C}$, where $\mathcal{X}$ is the set of all possible input phrases. 

There are many possible algorithms for text classification in natural language processing. In this project we will focus on the *Multinomial Naive Bayes* classifier.

### The MNB classifier
Given a class $c$ and a *document* (i.e. phrase) $d$, we can imagine that the document was composed by randomly extracting words from the total set of tokens $\mathcal{T} = \{t_1, t_2, \dots, t_m\}$. This way, the probability of composing the observed document would be

$P(d|c) = \prod_{1 \leq k \leq n_d} p(t_k|c)$.

Now, we want to infer $P(c|d)$. This can be done though Bayes's Theorem:

$P(c|d) \propto p(c) \prod_{1 \leq k \leq n_d} p(t_k|c)$.

We can then easily find the *maximum a posteriori* class $c_{map}$:

$c_{map} = \mathrm{argmax}_{c} P(d|c)$.

The maximum a posteriori class will be our guess for the classification.





### The learning algorithm
Our model depends on the following parameters:
* **Prior**: $p(c)$. 
* **Token probability**: $p(t_k|c)$, the probability for token $t_k$ to appear in a document of class $c$. 

The prior can be estimated as $\hat{p}(c) = \frac{N_c}{N}$, i.e. the fraction of documents of class $c$ in the training set.

The token probability can instead be estimated as $\hat{p}(t|c) = \frac{T_{ct} + 1}{\sum_{t'} (T_{ct' + 1})}$. The term $T_{ct}$ is the number of times token $t$ appears in a document of class $c$. Notice that, both at the numerator and denominator, we are adding $1$ to $T_{ct}$ and $T_{ct'}$. This is done in order to avoid having $p(t|c) = 0$ for tokens that never appear in documents of class $c$.

![Schema](/schema.png "Secord, 2002")


# Feature selection
Now that we have prepared our dataset, we want to select a subset of the possible tokens, choosing the ones that are most relevant for classification and eliminating *noise features*.

There are various quantities that can be computed to find the most useful tokens. We focused on the **mutual information** and on the $\chi^2$. 

The main idea is to obtain a ranking of the tokens (from "most useful" to "less useful") and use the first $k$ tokens for classification. The parameter $k$ will be varied to optimize the accuracy.

## Mutual information
The mutual information (MI) measures "how much the presence/absence of a term contributes to making the correct classification decision on the class" [1].

It is computed as

$\sum_{e_t\in\{0,1\}} \sum_{e_c\in\{0,...,n}} p(e_t, e_c)\log_2\frac{p(e_t,e_c)}{p(e_t)p(e_c)}$

We evaluated the MI for all possible tokens. The tokens were then ranked in order of decreasing MI.

Implementation:
```python3

#Evaluate MI for each token
for(i in 1:length(vocab.df$Token)){
    MI <- 0
    N.t <- sum(vocab.bool.df[i, 2:7])
    N.not.t <- n.records - N.t
    for(class in 0:5){
        MI.c <- 0
        N.c <- NCs[class + 1]
        N.not.c <- n.records - N.c 
        N.ct <- vocab.bool.df[i, class + 2]
        
        class.index <- switch(class + 1, "Class_0", "Class_1", 
                              "Class_2", "Class_3", "Class_4", "Class_5")
        N.not.c.not.t <- n.records - sum(vocab.bool.df[i, 2:7]) + N.ct - sum(vocab.bool.df$class.index)
        
        
        #P(class, token)log(...) + P(nonclass, token) + P(class, non token) + P(non class, non token)
        term <- ifelse(N.ct > 0, N.ct/n.records*log(N.ct*n.records/(N.c*N.t)), 0) + 
                ifelse((N.t - N.ct) > 0, (N.t - N.ct)/n.records*log((N.t-N.ct)*n.records/(N.not.c*N.t)), 0)+
                ifelse((N.c - N.ct) > 0, (N.c - N.ct)/n.records*log((N.c - N.ct)*n.records/(N.c * (N.not.t))), 0) + 
                ifelse(N.not.c.not.t > 0, 
                       N.not.c.not.t/n.records * log(N.not.c.not.t*n.records / (N.not.c * N.not.t)), 0)
                
        MI <- MI + term
        
    }
    
    MIs[i] <- MI
}
```

We get the following tokens (only displaying the first 12 for brevity):

|Token    |Class_0|Class_1|Class_2|Class_3|Class_4|Class_5|MI      |
|---------|-------|-------|-------|-------|-------|-------|--------|
|&lt;number&gt; |204    |609    |553    |974    |938    |729|1.152313|
|say      |233    |465    |442    |473    |398    |326    | 1.136849|
|state    | 55    |214    |163    |241    |253    |219    | 1.117582|
|percent  | 34    |115    |121    |228    |233    |193    | 1.116305|
|year     | 44    |155    |159    |230    |230    |184    | 1.115615|
|&lt;money>  | 54    |163    |184    |246    |228    |142    |1.114346|
|tax      | 60    |146    |164    |177    |175    |120    | 1.111194|
|obama    | 76    |130    |106    |131    | 86    | 68    | 1.110995|
|&lt;year>   | 43    |101    | 95    |153    |164    |152    | 1.110844|
|million  | 34    | 80    | 84    |140    |123    | 77    | 1.108695|
|president| 50    |116    | 92    |114    | 83    | 64    | 1.108282|
|s        | 48    |118    | 86    |115    |103    |107    | 1.108145|


## Chi-square method
The $\chi^2$ of two variables $A$ and $B$ is used to test their independence, i.e. to check if $P(A,B)$ is comparable to $P(A)P(B)$. 

In this context, we can compute the $\chi^2$ for the variables *token t appears in document d* and *document d belongs to class c*. For a fixed token, we can compute the $\chi^2$ for each class $c$ and perform an average across the classes.

```python3
for(i in 1:length(vocab.bool.df$Token)){
    Chi <- 0
    N.t <- sum(vocab.bool.df[i, 2:7])
    N.nott <- n.records - N.t 
    for(class in 0:5){
        N.c <- NCs[class + 1]
        N.ct <- vocab.bool.df[i, class + 2]
        E <- n.records * N.c/n.records * N.t/n.records
        Chi <- Chi + (N.ct - E)^2 / E
        
        # N_c,nontoken - E_c, nontoken 
        N.c.nott <- N.c - N.ct
        E <- n.records * N.c/n.records * N.nott/n.records
        Chi <- Chi + (N.c.nott - E)^2 / E
    }
    
    CHIs[i] <- Chi
}
```

The resulting rank is the following:

|Token    |Class_0|Class_1|Class_2|Class_3|Class_4|Class_5|Chi      |
|---------|-------|-------|-------|-------|-------|-------|---------|
|&lt;number> |204    |609    |553    |974    |938    |729    |106.61974|
|socialist| 10    |  1    |  1    |  0    |  0    |  1    | 84.35517|
|rep      | 30    | 19    | 25    | 16    |  5    | 10    | 77.77005|
|percent  | 34    |115    |121    |228    |233    |193    | 62.54439|
|say      |233    |465    |442    |473    |398    |326    | 43.91115|
|since    | 14    | 30    | 30    | 74    | 67    | 69    | 40.23848|
|obama    | 76    |130    |106    |131    | 86    | 68    | 39.15072|
|duffy    |  4    |  1    |  0    |  0    |  0    |  0    | 35.93560|
|cut      | 12    | 44    | 68    | 99    | 72    | 40    | 35.71443|
|cup      |  3    |  0    |  0    |  0    |  0    |  0    | 34.40530|
|advisory |  3    |  0    |  0    |  0    |  0    |  0    | 34.40530|
|sic      |  3    |  0    |  0    |  0    |  0    |  0    | 34.40530|



The ranking seems less convincing this time: we notice that words which appear in only one class rank high on the list. This can be a source of error. Furthermore, words which intuitively convey less meaning. We therefore choose to use the token ranking provided by the MI.

# Training

We now vary the number of tokens $k$ and train the classifier on those tokens. In the following graph we plot the accuracy obtained when predicting the labels of the validation set.

<div style="display: flex; justify-content: center;">
  <img src="/images/scores.png" alt="Alt Text" width="500px" height="500px" />
</div>

The vocabulary size with optimum accuracy is $708$, with a score of $24\%$. This results could be explained by the fact that our classification has to detect different "shades" of truth, which might be too delicate of a task for a Naive Bayes Classifier. 

Indeed, when looking at the confusion matrix, one notices that the classifier tends to swap classes which are "close" to one another, e.g. FALSE and BARELY TRUE.


<div style="display: flex; justify-content: center;">
  <img src="/images/confusion.png" alt="Alt Text" width="500px" height="500px" />
</div>


