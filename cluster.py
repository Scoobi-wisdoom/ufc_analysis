import sqlalchemy
import pandas as pd
import pymysql

path = '../scraper/'
# MYSQL 에 연결
with open(path + "db_name.txt", "r") as f:
    lines = f.readlines()
    pw = lines[0].strip()
    db = lines[1].strip()
engine = sqlalchemy.create_engine("mysql+pymysql://{user}:{pw}@localhost/{db}".format(user="root", pw=pw, db=db))

# 1. 데이터베이스에서 Table 을 불러온다.
## locations 정보를 불러온다. sql
with engine.connect() as con:
    locations = pd.read_sql_table('locations', con=con)

## events 정보를 불러온다. sql
with engine.connect() as con:
    events = pd.read_sql_table('events', con=con)

## weights Table 을 MYSQL 에서 불러온다.
with engine.connect() as con:
    weights = pd.read_sql_table('weights', con)

## methods Table 을 MYSQL 에서 불러온다.
with engine.connect() as con:
    methods = pd.read_sql_table('methods', con)

## times Table 을 MYSQL 에서 불러온다.
with engine.connect() as con:
    times = pd.read_sql_table('times', con)

## results Table 을 MYSQL 에서 불러온다.
with engine.connect() as con:
    results = pd.read_sql_table('results', con)

## referees Table 을 MYSQL 에서 불러온다.
with engine.connect() as con:
    referees = pd.read_sql_table('referees', con)

## achieves Table 을 MYSQL 에서 불러온다.
with engine.connect() as con:
    achieves = pd.read_sql_table('achieves', con)

## fighters Table 을 MYSQL 에서 불러온다.
with engine.connect() as con:
    fighters = pd.read_sql_table('fighters', con=con)

## matches Table 을 MYSQL 에서 불러온다.
with engine.connect() as con:
    matches = pd.read_sql_table('matches', con=con)

## achievematches Table 을 MYSQL 에서 불러온다.
with engine.connect() as con:
    achievematches = pd.read_sql_table('achievematches', con=con)

## rounds Table 을 MYSQL 에서 불러온다.
with engine.connect() as con:
    rounds = pd.read_sql_table('rounds', con=con)

# 2. 초 단위 경기 시간 (데이터 전처리)
## time_format 이 3 Rnd (5-5-5) 와 5 Rnd (5-5-5-5-5) 인 것만을 대상. time_id = [16, 17]. 96% 비중. 그런데 [18] 은 뭐지?
## [18] 역시 3라운드 각 5분이기 때문에 포함하는 것이 좋다.
target_match_id_list = matches[matches['time_id'].isin([16, 17, 18])]['match_id'].to_list()

## 지정된 time_format 에 해당하는 rounds Table 을 불러온다. 마지막 라운드가 몇 라운드인지 last_round 로 표시한다.
rounds_of_target = rounds[rounds['match_id'].isin(target_match_id_list)]
last_round = rounds_of_target.loc[rounds_of_target.groupby('match_id')['round_number'].idxmax()]['round_number']

## 각 match_id 별로 경기 시간을 초 단위로 표시한다. 1 라운드 당 5분.
min_to_sec = (last_round.copy() - 1) * 5 * 60
min_to_sec.reset_index(drop=True, inplace=True)
sec = matches.loc[target_match_id_list]['time_second']
sec.reset_index(drop=True, inplace=True)
## match_id 와 seconds 로 구성된 dataframe
match_id_sec = pd.DataFrame({'match_id': target_match_id_list, 'seconds': min_to_sec + sec})

## grappling, strikes 는 총 경기 시간으로 나눈 값을 적용해야 한다.
round_sum_by_fighter = rounds.drop('round_number', axis=1).groupby(['match_id', 'fighter_id'], as_index=False).sum()
round_sum_by_fighter.reset_index(drop=True, inplace=True)
(list(set(round_sum_by_fighter['match_id'])).sort == list(set(match_id_sec['match_id'])).sort())

fighters[fighters['fighter_name'].str.contains('Khabib')][['fighter_id', 'fighter_name']]