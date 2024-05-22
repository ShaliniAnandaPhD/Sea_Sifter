import random
import string
import datetime
from pymongo import MongoClient

class CitizenScienceGamification:
    def __init__(self, db_connection_string):
        """
        Initialize the CitizenScienceGamification module.

        :param db_connection_string: Connection string for the MongoDB database.
        """
        self.client = MongoClient(db_connection_string)
        self.db = self.client['seasifter']
        self.users_collection = self.db['users']
        self.samples_collection = self.db['samples']
        self.challenges_collection = self.db['challenges']
        self.rewards_collection = self.db['rewards']

    def register_user(self, username, email, password):
        """
        Register a new user in the gamification system.

        :param username: Username of the user.
        :param email: Email address of the user.
        :param password: Password for the user account.

        Possible errors:
        - Duplicate username or email.
        - Invalid email format.
        - Weak password.

        Solutions:
        - Check for existing username or email before registration.
        - Validate email format using regular expressions or email validation libraries.
        - Enforce password strength requirements (e.g., minimum length, character diversity).
        """
        try:
            # Check if username or email already exists
            existing_user = self.users_collection.find_one({'$or': [{'username': username}, {'email': email}]})
            if existing_user:
                raise ValueError("Username or email already exists.")

            # Validate email format (simple check, use a more robust validation in production)
            if '@' not in email:
                raise ValueError("Invalid email format.")

            # Enforce password strength requirements (e.g., minimum length, character diversity)
            if len(password) < 8:
                raise ValueError("Password must be at least 8 characters long.")

            # Create a new user document
            user = {
                'username': username,
                'email': email,
                'password': password,  # Hash the password before storing (not shown here)
                'registered_at': datetime.datetime.now(),
                'points': 0,
                'badges': [],
                'completed_challenges': []
            }
            self.users_collection.insert_one(user)
            print(f"User '{username}' registered successfully.")
        except ValueError as ve:
            print(f"Error: {str(ve)}")
        except Exception as e:
            print(f"An error occurred during user registration: {str(e)}")

    def login_user(self, username, password):
        """
        Authenticate and log in a user.

        :param username: Username of the user.
        :param password: Password for the user account.

        Possible errors:
        - User not found.
        - Incorrect password.

        Solutions:
        - Handle user not found error and provide appropriate feedback.
        - Verify password hash instead of storing plaintext passwords.
        """
        try:
            user = self.users_collection.find_one({'username': username})
            if not user:
                raise ValueError("User not found.")

            # Verify password (compare hash instead of plaintext)
            if password != user['password']:
                raise ValueError("Incorrect password.")

            print(f"User '{username}' logged in successfully.")
            return user
        except ValueError as ve:
            print(f"Error: {str(ve)}")
        except Exception as e:
            print(f"An error occurred during user login: {str(e)}")

    def submit_sample(self, user_id, sample_data):
        """
        Submit a microplastic sample collected by a user.

        :param user_id: ID of the user submitting the sample.
        :param sample_data: Data of the collected microplastic sample.

        Possible errors:
        - Invalid user ID.
        - Missing required sample data fields.

        Solutions:
        - Validate user ID before processing the sample submission.
        - Ensure all required sample data fields are present.
        """
        try:
            user = self.users_collection.find_one({'_id': user_id})
            if not user:
                raise ValueError("Invalid user ID.")

            # Validate sample data fields
            required_fields = ['location', 'date', 'type', 'quantity']
            for field in required_fields:
                if field not in sample_data:
                    raise ValueError(f"Missing required field: {field}")

            sample_data['user_id'] = user_id
            sample_data['submitted_at'] = datetime.datetime.now()
            self.samples_collection.insert_one(sample_data)
            print(f"Sample submitted successfully by user '{user['username']}'.")

            # Award points or badges based on sample submission
            self.award_points(user_id, 10)  # Example: Award 10 points for each sample submission
            self.check_badge_criteria(user_id)  # Check if the user qualifies for any badges
        except ValueError as ve:
            print(f"Error: {str(ve)}")
        except Exception as e:
            print(f"An error occurred during sample submission: {str(e)}")

    def create_challenge(self, challenge_data):
        """
        Create a new challenge for citizen scientists.

        :param challenge_data: Data of the challenge.

        Possible errors:
        - Missing required challenge data fields.
        - Invalid challenge data format.

        Solutions:
        - Ensure all required challenge data fields are present.
        - Validate challenge data format and handle any invalid data.
        """
        try:
            # Validate challenge data fields
            required_fields = ['title', 'description', 'start_date', 'end_date', 'reward_points']
            for field in required_fields:
                if field not in challenge_data:
                    raise ValueError(f"Missing required field: {field}")

            challenge_data['created_at'] = datetime.datetime.now()
            self.challenges_collection.insert_one(challenge_data)
            print(f"Challenge '{challenge_data['title']}' created successfully.")
        except ValueError as ve:
            print(f"Error: {str(ve)}")
        except Exception as e:
            print(f"An error occurred during challenge creation: {str(e)}")

    def complete_challenge(self, user_id, challenge_id):
        """
        Mark a challenge as completed by a user.

        :param user_id: ID of the user completing the challenge.
        :param challenge_id: ID of the challenge being completed.

        Possible errors:
        - Invalid user ID or challenge ID.
        - Challenge already completed by the user.

        Solutions:
        - Validate user ID and challenge ID before processing the completion.
        - Check if the user has already completed the challenge.
        """
        try:
            user = self.users_collection.find_one({'_id': user_id})
            if not user:
                raise ValueError("Invalid user ID.")

            challenge = self.challenges_collection.find_one({'_id': challenge_id})
            if not challenge:
                raise ValueError("Invalid challenge ID.")

            if challenge_id in user['completed_challenges']:
                raise ValueError("Challenge already completed by the user.")

            self.users_collection.update_one(
                {'_id': user_id},
                {'$push': {'completed_challenges': challenge_id}}
            )
            print(f"Challenge '{challenge['title']}' completed by user '{user['username']}'.")

            # Award points or badges based on challenge completion
            self.award_points(user_id, challenge['reward_points'])
            self.check_badge_criteria(user_id)
        except ValueError as ve:
            print(f"Error: {str(ve)}")
        except Exception as e:
            print(f"An error occurred during challenge completion: {str(e)}")

    def award_points(self, user_id, points):
        """
        Award points to a user.

        :param user_id: ID of the user being awarded points.
        :param points: Number of points to award.

        Possible errors:
        - Invalid user ID.
        - Negative points value.

        Solutions:
        - Validate user ID before awarding points.
        - Ensure points value is non-negative.
        """
        try:
            user = self.users_collection.find_one({'_id': user_id})
            if not user:
                raise ValueError("Invalid user ID.")

            if points < 0:
                raise ValueError("Points must be a non-negative value.")

            self.users_collection.update_one(
                {'_id': user_id},
                {'$inc': {'points': points}}
            )
            print(f"{points} points awarded to user '{user['username']}'.")
        except ValueError as ve:
            print(f"Error: {str(ve)}")
        except Exception as e:
            print(f"An error occurred during points awarding: {str(e)}")

    def create_badge(self, badge_data):
        """
        Create a new badge for citizen scientists.

        :param badge_data: Data of the badge.

        Possible errors:
        - Missing required badge data fields.
        - Invalid badge data format.

        Solutions:
        - Ensure all required badge data fields are present.
        - Validate badge data format and handle any invalid data.
        """
        try:
            # Validate badge data fields
            required_fields = ['name', 'description', 'criteria']
            for field in required_fields:
                if field not in badge_data:
                    raise ValueError(f"Missing required field: {field}")

            badge_data['created_at'] = datetime.datetime.now()
            self.rewards_collection.insert_one(badge_data)
            print(f"Badge '{badge_data['name']}' created successfully.")
        except ValueError as ve:
            print(f"Error: {str(ve)}")
        except Exception as e:
            print(f"An error occurred during badge creation: {str(e)}")

    def check_badge_criteria(self, user_id):
        """
        Check if a user meets the criteria for any badges and award them.

        :param user_id: ID of the user to check badge criteria for.

        Possible errors:
        - Invalid user ID.
        - Error retrieving user data or badge criteria.

        Solutions:
        - Validate user ID before checking badge criteria.
        - Handle errors that may occur during data retrieval.
        """
        try:
            user = self.users_collection.find_one({'_id': user_id})
            if not user:
                raise ValueError("Invalid user ID.")

            badges = self.rewards_collection.find()
            for badge in badges:
                if badge['_id'] not in user['badges']:
                    criteria_met = self.evaluate_badge_criteria(user, badge['criteria'])
                    if criteria_met:
                        self.users_collection.update_one(
                            {'_id': user_id},
                            {'$push': {'badges': badge['_id']}}
                        )
                        print(f"Badge '{badge['name']}' awarded to user '{user['username']}'.")
        except ValueError as ve:
            print(f"Error: {str(ve)}")
        except Exception as e:
            print(f"An error occurred during badge criteria check: {str(e)}")

    def evaluate_badge_criteria(self, user, criteria):
        """
        Evaluate if a user meets the specified badge criteria.

        :param user: User data.
        :param criteria: Badge criteria to evaluate.

        Possible errors:
        - Invalid criteria format.
        - Error retrieving user data or sample data.

        Solutions:
        - Validate criteria format and handle any invalid criteria.
        - Handle errors that may occur during data retrieval.
        """
        try:
            # Example criteria evaluation (customize based on your specific criteria format)
            if criteria['type'] == 'sample_count':
                count = self.samples_collection.count_documents({'user_id': user['_id']})
                return count >= criteria['value']
            elif criteria['type'] == 'challenge_count':
                count = len(user['completed_challenges'])
                return count >= criteria['value']
            else:
                raise ValueError(f"Invalid badge criteria type: {criteria['type']}")
        except ValueError as ve:
            print(f"Error: {str(ve)}")
            return False
        except Exception as e:
            print(f"An error occurred during badge criteria evaluation: {str(e)}")
            return False

    def get_leaderboard(self, limit=10):
        """
        Get the leaderboard of top citizen scientists based on points.

        :param limit: Maximum number of users to include in the leaderboard.

        Possible errors:
        - Invalid limit value.
        - Error retrieving user data.

        Solutions:
        - Validate limit value and handle any invalid input.
        - Handle errors that may occur during data retrieval.
        """
        try:
            if limit <= 0:
                raise ValueError("Leaderboard limit must be a positive integer.")

            leaderboard = self.users_collection.find().sort('points', -1).limit(limit)
            return list(leaderboard)
        except ValueError as ve:
            print(f"Error: {str(ve)}")
            return []
        except Exception as e:
            print(f"An error occurred while retrieving the leaderboard: {str(e)}")
            return []

# Example usage
def main():
    # MongoDB connection string
    db_connection_string = 'mongodb://localhost:27017'

    # Create an instance of CitizenScienceGamification
    gamification = CitizenScienceGamification(db_connection_string)

    # Register a new user
    gamification.register_user('johndoe', 'johndoe@example.com', 'password123')

    # Login user
    user = gamification.login_user('johndoe', 'password123')

    # Submit a sample
    sample_data = {
        'location': 'Beach A',
        'date': datetime.datetime.now(),
        'type': 'microplastic',
        'quantity': 10
    }
    gamification.submit_sample(user['_id'], sample_data)

    # Create a challenge
    challenge_data = {
        'title': 'Monthly Sampling Challenge',
        'description': 'Collect 50 samples in a month',
        'start_date': datetime.datetime.now(),
        'end_date': datetime.datetime.now() + datetime.timedelta(days=30),
        'reward_points': 100
    }
    gamification.create_challenge(challenge_data)

    # Complete a challenge
    challenges = gamification.challenges_collection.find()
    challenge_id = challenges[0]['_id']
    gamification.complete_challenge(user['_id'], challenge_id)

    # Create a badge
    badge_data = {
        'name': 'Sampling Enthusiast',
        'description': 'Collected 100 samples',
        'criteria': {
            'type': 'sample_count',
            'value': 100
        }
    }
    gamification.create_badge(badge_data)

    # Get the leaderboard
    leaderboard = gamification.get_leaderboard()
    print("Leaderboard:")
    for rank, user in enumerate(leaderboard, start=1):
        print(f"{rank}. {user['username']} - Points: {user['points']}")

if __name__ == '__main__':
    main()
