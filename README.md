# Assignment 1

[https://www.kaggle.com/uciml/pima-indians-diabetes-database](link)

1.Import relevant commands for numpy, pandas, sklearn.

```python
import numpy as np
import pandas as pd
from pandas import DataFrame, read_csv, to_numeric
from sklearn.model_selection import train_test_split, KFold
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
import math
import scipy.stats as stats
```
2.Using the appropriate pandas function, read the diabetes.csv into a dataframe. Pay good attention to the necessary arguments.
```python
data = read_csv('diabetes.csv', sep=",")
a = pd.DataFrame(data)
print(a)<br/><br/>
```

```Output
     Pregnancies  Glucose  BloodPressure  SkinThickness  Insulin   BMI  \
0              6      148             72             35        0  33.6
1              1       85             66             29        0  26.6
2              8      183             64              0        0  23.3
3              1       89             66             23       94  28.1
4              0      137             40             35      168  43.1
..           ...      ...            ...            ...      ...   ...
763           10      101             76             48      180  32.9
764            2      122             70             27        0  36.8
765            5      121             72             23      112  26.2
766            1      126             60              0        0  30.1
767            1       93             70             31        0  30.4

     DiabetesPedigreeFunction  Age  Outcome
0                       0.627   50        1
1                       0.351   31        0
2                       0.672   32        1
3                       0.167   21        0
4                       2.288   33        1
..                        ...  ...      ...
763                     0.171   63        0
764                     0.340   27        0
765                     0.245   30        0
766                     0.349   47        1
767                     0.315   23        0

[768 rows x 9 columns]

```
3.use naivebayes, logistic regression and 3-nn classifiers (library) to train on the training sets and compute training and validation errors for each fold. The target label is Outcome.
```python
X=a.iloc[:,0:8].values
y=a.iloc[:,-1].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)
kf = KFold(n_splits=10)
kf.get_n_splits()
#print(kf)
for train_index, test_index in kf.split(X):
    print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

nb_model = GaussianNB()
lr_model = LogisticRegression(solver= 'liblinear')
knn_model = KNeighborsClassifier(n_neighbors=3)

nb_error = []
lr_error = []
knn_error =[]

for train, validation in kf.split(X_train):
    nb_fit = nb_model.fit(X_train[train], y_train[train])
    nb_pred = nb_model.predict(X_train[validation])
    #print(nb_pred)
    error = np.sum(nb_pred!= y_train[validation])/len(nb_pred)
    nb_error.append(error)

    lr_fit = lr_model.fit(X_train[train], y_train[train])
    lr_pred = lr_model.predict(X_train[validation])
    #print(lr_pred)
    error = np.sum(lr_pred!=y_train[validation])/len(lr_pred)
    lr_error.append(error)

    knn_fit = knn_model.fit(X_train[train], y_train[train])
    knn_pred = knn_model.predict(X_train[validation])
    #print(knn_pred)
    error = np.sum(lr_pred!=y_train[validation])/len(knn_pred)
    knn_error.append(error)

Error_list=np.transpose([nb_error,lr_error,knn_error])
z = pd.DataFrame(data=Error_list, columns =("Naivebayes","logistic regression","KNN"))
print(z)
```
```output
TRAIN: [ 77  78  79  80  81  82  83  84  85  86  87  88  89  90  91  92  93  94
  95  96  97  98  99 100 101 102 103 104 105 106 107 108 109 110 111 112
 113 114 115 116 117 118 119 120 121 122 123 124 125 126 127 128 129 130
 131 132 133 134 135 136 137 138 139 140 141 142 143 144 145 146 147 148
 149 150 151 152 153 154 155 156 157 158 159 160 161 162 163 164 165 166
 167 168 169 170 171 172 173 174 175 176 177 178 179 180 181 182 183 184
 185 186 187 188 189 190 191 192 193 194 195 196 197 198 199 200 201 202
 203 204 205 206 207 208 209 210 211 212 213 214 215 216 217 218 219 220
 221 222 223 224 225 226 227 228 229 230 231 232 233 234 235 236 237 238
 239 240 241 242 243 244 245 246 247 248 249 250 251 252 253 254 255 256
 257 258 259 260 261 262 263 264 265 266 267 268 269 270 271 272 273 274
 275 276 277 278 279 280 281 282 283 284 285 286 287 288 289 290 291 292
 293 294 295 296 297 298 299 300 301 302 303 304 305 306 307 308 309 310
 311 312 313 314 315 316 317 318 319 320 321 322 323 324 325 326 327 328
 329 330 331 332 333 334 335 336 337 338 339 340 341 342 343 344 345 346
 347 348 349 350 351 352 353 354 355 356 357 358 359 360 361 362 363 364
 365 366 367 368 369 370 371 372 373 374 375 376 377 378 379 380 381 382
 383 384 385 386 387 388 389 390 391 392 393 394 395 396 397 398 399 400
 401 402 403 404 405 406 407 408 409 410 411 412 413 414 415 416 417 418
 419 420 421 422 423 424 425 426 427 428 429 430 431 432 433 434 435 436
 437 438 439 440 441 442 443 444 445 446 447 448 449 450 451 452 453 454
 455 456 457 458 459 460 461 462 463 464 465 466 467 468 469 470 471 472
 473 474 475 476 477 478 479 480 481 482 483 484 485 486 487 488 489 490
 491 492 493 494 495 496 497 498 499 500 501 502 503 504 505 506 507 508
 509 510 511 512 513 514 515 516 517 518 519 520 521 522 523 524 525 526
 527 528 529 530 531 532 533 534 535 536 537 538 539 540 541 542 543 544
 545 546 547 548 549 550 551 552 553 554 555 556 557 558 559 560 561 562
 563 564 565 566 567 568 569 570 571 572 573 574 575 576 577 578 579 580
 581 582 583 584 585 586 587 588 589 590 591 592 593 594 595 596 597 598
 599 600 601 602 603 604 605 606 607 608 609 610 611 612 613 614 615 616
 617 618 619 620 621 622 623 624 625 626 627 628 629 630 631 632 633 634
 635 636 637 638 639 640 641 642 643 644 645 646 647 648 649 650 651 652
 653 654 655 656 657 658 659 660 661 662 663 664 665 666 667 668 669 670
 671 672 673 674 675 676 677 678 679 680 681 682 683 684 685 686 687 688
 689 690 691 692 693 694 695 696 697 698 699 700 701 702 703 704 705 706
 707 708 709 710 711 712 713 714 715 716 717 718 719 720 721 722 723 724
 725 726 727 728 729 730 731 732 733 734 735 736 737 738 739 740 741 742
 743 744 745 746 747 748 749 750 751 752 753 754 755 756 757 758 759 760
 761 762 763 764 765 766 767] TEST: [ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23
 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47
 48 49 50 51 52 53 54 55 56 57 58 59 60 61 62 63 64 65 66 67 68 69 70 71
 72 73 74 75 76]

   Naivebayes  logistic regression       KNN
0    0.328571             0.314286  0.314286
1    0.157143             0.171429  0.171429
2    0.231884             0.188406  0.188406
3    0.333333             0.304348  0.304348
4    0.304348             0.333333  0.333333
5    0.217391             0.202899  0.202899
6    0.173913             0.159420  0.159420
7    0.202899             0.188406  0.188406
8    0.202899             0.159420  0.159420
9    0.246377             0.260870  0.260870
```
4.is the error of naive bayes <0.2 with confidence 0.9
```python
x=np.std(nb_error)
y=np.mean(nb_error)
z= math.sqrt(10)*(y-0.2)/x
print(z)
a=stats.t.ppf(0.9,9)
if z > a :
    print("No")
else :
    print("Yes")
```
```Output
2.122373130045732
No
```

5.have naive bayes and knn the same error?
```python
x=np.absolute(np.subtract(np.array(knn_error),np.array(nb_error)))
print(x)
y=np.std(x)
z=np.mean(x)
a= math.sqrt(10)*z/y
print(a)
b=stats.t.ppf(0.9,9)
if a > b :
    print("No")
else :
    print("Yes")
 ```
 ```Output
[0.01428571 0.01428571 0.04347826 0.02898551 0.02898551 0.01449275
 0.01449275 0.01449275 0.04347826 0.01449275]
6.296258830957976
No
```
6.do the three classifiers have different errors?
```python
e_table = np.array(np.transpose([nb_error,lr_error,knn_error]))
No_of_models = 3
Folds= 10
average_of_models =np.mean(e_table, axis=0)
bet_same_model = Folds*np.var(average_of_models)
bet_all_model = np.sum(np.var(e_table,axis=0)/No_of_models)
x=stats.f.ppf(0.9,dfn=No_of_models-1 ,dfd= No_of_models*(Folds-1))
ratio= bet_same_model/bet_all_model
print("F-statistic",x)
print("Estimater ratio",ratio)
if x < ratio:
    print("All models have same error")
else:
    print("All models have differnt error")
```
```Output
F-statistic 2.5106086665585408
Estimater ratio 0.0753450085421447
All models have differnt error
```

7.Use Bayes rule to decide on the label using the Glucose feature. Compute the mean and std of Glucose feature on the whole dataset (marginal distribution) and for each class separately, (class condition distribution Prob(x|C=0), Prob(x|C=1))). Assume that the feature is distributed according to the mean and std you computed. Compute the predictions for the 10 validation sets. Compare with the naivebayes classifier (library) using only the Glucose feature
```python
from sklearn.naive_bayes import GaussianNB
data_g = data.Glucose.values
data_o = data.Outcome.values

err_model = []
err_lib = []

def Gaussian(x,mean,var):
    numerator = np.exp(-((x-mean )**2)/(2*var))
    denominator = np.sqrt(2*np.pi*var)
    return numerator/denominator

for train, validation in kf.split(data_o):
    x_train = data_g[train]
    y_train = data_o[train]

    var_0 = np.var(x_train[y_train==0])
    var_1 = np.var(x_train[y_train==1])

    p_prob_0 = len(x_train[y_train==0]/len(x_train))
    p_prob_1 = len(x_train[y_train==1]/len(x_train))

    mean_0 = np.mean(x_train[y_train==0])
    mean_1 = np.mean(x_train[y_train==1])

    prediction= []

    for x in data_g[validation]:
        gaussian_0= Gaussian(x,mean_0,var_0)
        gaussian_1= Gaussian(x,mean_1,var_1)

        class_0 = p_prob_0*gaussian_0
        class_1 = p_prob_1*gaussian_1

        if class_0 > class_1:
            prediction.append(0)
        else:
            prediction.append(1)

    err_model.append(np.sum(np.array(prediction)!=data_o[validation])/len(prediction))

    nb_classifier = GaussianNB()
    nb_classifier.fit(data_g.reshape(-1,1), data_o.reshape(-1,1))
    nb_pred = nb_classifier.predict(data_g[validation].reshape(-1,1))
    error = np.sum(nb_pred!=data_o[validation])/len(data_g[validation])
    err_lib.append(error)

Error_list = np.transpose([err_model, err_lib])
z= pd.DataFrame(data=Error_list , columns =("My model Error","Library Model error"))
print("\n",z)
print("\n Both model have same error\n")
```
```Output
    My model Error  Library Model error
0        0.337662             0.337662
1        0.246753             0.246753
2        0.298701             0.298701
3        0.324675             0.324675
4        0.246753             0.246753
5        0.220779             0.194805
6        0.181818             0.181818
7        0.207792             0.207792
8        0.210526             0.210526
9        0.250000             0.250000

 Both model have same error
 ```
