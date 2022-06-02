# Ensemble Of Segmentation
This repository contains the code used in the research paper: <h3><a href="https://www.mdpi.com/1659504"> An empirical study on ensemble of segmentation approaches </a></h3>
<h4> Authors: </h4> Loris Nanni, Alessandra Lumini, Andrea Loreggia, Alberto Formaggio and Daniela Cuza
<br>
<hr>
I was part of the research team of this work. <br>
In particular, my contribution is related to the realization of new loss functions. Several of the aforementioned loss functions were used in the ensemble in order to improve the diversity of the classifier and achieve higher accuracy.
<br><br>
In the folder <a href="/LossFunctions">LossFunctions</a> you can find the loss functions that were used in the ensembles after testing which ones performed better on a ResNet-10 (I had to use ResNet-10 due to the limited computing power available in the computer I was using for debugging).<br><br>
However, I developed more loss than the ones used in the paper (discarded due to the lower accuracy they had compared to the other ones).<br>
All the loss that were developed can be found in the folder <a href="/MyLossFunctions">MyLossFunctions</a>
