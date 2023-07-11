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

<hr style="border:1px solid gray">
