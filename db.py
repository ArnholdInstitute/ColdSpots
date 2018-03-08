import psycopg2, os
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

db_args = {
    'dbname' : 'aigh',
    'host' : os.environ.get('PGHOST', 'localhost'),
    'user' : os.environ.get('PGUSER', ''),
    'password' : os.environ.get('PGPASSWORD', '')
}
conn = psycopg2.connect(**db_args)