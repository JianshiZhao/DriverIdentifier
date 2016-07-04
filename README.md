# DriverIdentifier
This project is to identify drivers behavior from images and classify them.

Running Results:

Total dataset size:

n_samples: 22424

n_features: 19200

n_classes: 10

Extracting the top 50 eigen_images from 19200 training images

done in 38.434s

Projecting the input data on the eigen_images orthonormal basis

done in 5.301s

Fitting the classifier to the training set

done in 171.596s

Best estimator found by grid search:

SVC(C=50, cache_size=200, class_weight='balanced', coef0=0.0,

  decision_function_shape=None, degree=3, gamma=5e-08, kernel='rbf',

  max_iter=-1, probability=True, random_state=None, shrinking=True,

  tol=0.001, verbose=False)

Predict driver behavior classes: 

done in 10.913s

Prediction Accuracy: 0.993399928648
