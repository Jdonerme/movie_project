import matplotlib.pyplot as plt
import numpy as np
import util as ut

# Get the rating information
ratings_array, rating_info = ut.get_categories()

# Generate histogram of all ratings in the dataset
plt.hist(ratings_array, bins=np.arange(0.5, 6.5, 1))
plt.title('Histogram of All Ratings')
plt.xlabel('Ratings')
plt.savefig('histograms/4.1.png')
plt.show()

# Generate histogram of ratings of 10 best movies
plt.hist(rating_info[0], bins=np.arange(0.5, 6.5, 1))
plt.title('Histogram of Ratings of 10 Best Movies')
plt.xlabel('Ratings')
plt.savefig('histograms/4.3.png')
plt.show()

# Generate histogram of ratings of 10 most popular movies
plt.hist(rating_info[1], bins=np.arange(0.5, 6.5, 1))
plt.title('Histogram of Ratings of 10 Most Popular Movies')
plt.xlabel('Ratings')
plt.savefig('histograms/4.2.png')
plt.show()

# Generate histogram of ratings of Animation movies
plt.hist(rating_info[2], bins=np.arange(0.5, 6.5, 1))
plt.title('Histogram of Ratings of Animation Movies')
plt.xlabel('Ratings')
plt.savefig('histograms/4.4anim.png')
plt.show()

# Generate histogram of ratings of Comedy movies
plt.hist(rating_info[3], bins=np.arange(0.5, 6.5, 1))
plt.title('Histogram of Ratings of Comedy Movies')
plt.xlabel('Ratings')
plt.savefig('histograms/4.4com.png')
plt.show()

# Generate histogram of ratings of Musical movies
plt.hist(rating_info[4], bins=np.arange(0.5, 6.5, 1))
plt.title('Histogram of Ratings of Musical Movies')
plt.xlabel('Ratings')
plt.savefig('histograms/4.4mus.png')
plt.show()