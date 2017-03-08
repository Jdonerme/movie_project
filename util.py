GENRE_MAP = {'Unknown' : 0, 'Action' : 1, 'Adventure' : 2, 'Animation' : 3, \
'Childrens' : 4, 'Comedy' : 5, 'Crime': 6, 'Documentary' : 7, 'Drama' : 8, \
'Fantasy' : 9, 'Film-Noir' : 10, 'Horror' : 11, 'Musical' : 12, 'Mystery' : 13, \
'Romance' : 14, 'Sci-Fi' : 15, 'Thriller' : 16, 'War' : 17, 'Western' : 18}

""" Gets data from movie.txt file.
    
    Imports data. Returns a dictionary mapping the movie ids to the title
    as well as a list of movie titles in each genre specified.
    Only handles exactly 3 genres.
    

    """
def get_movie_data(genres_desired):
    assert len(genres_desired) == 3
    movies = []
    titles = {}
    genre_lists= ([], [], [])
    with open("movies.txt") as f:
        for line in f:
            data = line.strip("\n").split("\t")
            mov_id = int(data[0])
            movies.append(mov_id)
            titles[mov_id] = data[1]
            genre_list = data[2:]
            for i in range(len(genres_desired)):
                g = genres_desired[i]
                if int(genre_list[GENRE_MAP[g]]) == 1:
                    genre_lists[i].append(mov_id)
    return movies, titles, genre_lists
    
""" Gets Rating Dta from importing from the data file 
    
    """
def get_rating_data(genres_desired):
    avg_ratings = {}
    num_ratings = {}
    movies, _, _ = get_movie_data(genres_desired)
    genre_ratings = ([], [], [])
    ratings_array = []
    data_array = []
    for mov in movies:
        num_ratings[mov] = 0.0
        avg_ratings[mov] = 0.0
    with open("data.txt") as f:
        for line in f:
            data = line.strip("\n").split("\t")
            mov_id = int(data[1])
            rating = float(data[2])
            # update how rated the movie is
            num_ratings[mov_id] += 1
            # update its average rating
            new_rating = (num_ratings[mov_id] - 1.0) * avg_ratings[mov_id] + rating
            new_rating /= num_ratings[mov_id]
            avg_ratings[mov_id] = new_rating
            
            # update array of all ratings
            ratings_array.append(int(data[2]))
            temp = (int(data[0]), int(data[1]), int(data[2]))
        
           
            # update array we'll use for matrix factorization
            data_array.append(temp)
    return data_array, ratings_array, num_ratings, avg_ratings

""" Gets a list of all occurances of ratings for the top rated, most rated,
    and genre specific movies.
    
    Each ratings list is in a seperate element of the tuple
    in order of top, most popular, genres. 
    
    """
def get_ratings_from_ids(top, pop, genres):
    ratings = ([], [], [], [], [])
    with open("data.txt") as f:
        for line in f:
            data = line.strip("\n").split("\t")
            mov_id = int(data[1])
            rating = float(data[2])
            if mov_id in top:
                ratings[0].append(rating)
            if mov_id in pop:
                ratings[1].append(rating)
            if mov_id in genres[0]:
                ratings[2].append(rating)
            if mov_id in genres[1]:
                ratings[3].append(rating)
            if mov_id in genres[2]:
                ratings[4].append(rating)
    return ratings
    
""" Imports data and returns lists of ratings of data based on movies
    that fall in to categories specified on the assignment.
    
    Returns 2 elements. The first is all of the ratings in an array. The 
    second is a tuple. The elements in the tuple contain lists of all movie 
    ratings that fit criteria that match the get_ratings_from_id function.
    i.e. in order of top, most popular, genres. 
    
    """
def get_categories(genres_desired=['Animation', 'Comedy', 'Musical']):
    movies, _, genre_lists = get_movie_data(genres_desired)
    data_array, ratings_array, num_ratings, avg_ratings = get_rating_data(genres_desired)
    top = sorted(movies, key=avg_ratings.get, reverse=True)[:10]
    pop = sorted(movies, key=num_ratings.get, reverse=True)[:10]

    rating_info = get_ratings_from_ids(top, pop, genre_lists)
    return ratings_array, rating_info
