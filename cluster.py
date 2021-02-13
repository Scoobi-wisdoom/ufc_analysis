import sqlalchemy
import pandas as pd
import pymysql
import matplotlib.pyplot as plt
import numpy as np
from sklearn import preprocessing
from sklearn.cluster import KMeans

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
## rounds 데이터를 match_id 와 fighter_id 를 기준으로 그룹화한 후에 모두 sum() 한다.
round_sum_by_fighter = rounds.drop('round_number', axis=1).groupby(['match_id', 'fighter_id'], as_index=False).sum()
round_max_by_fighter = rounds[['match_id', 'fighter_id', 'round_number']].groupby(['match_id', 'fighter_id'], as_index=False).max()
round_sum_by_fighter.reset_index(drop=True, inplace=True)

## 총 승리 통계. result_id 0 WL, result_id 3 LW.
target_matches = matches.loc[target_match_id_list]
total_red_win = target_matches[target_matches['result_id'] == 0][['match_id', 'fighter_red_id']]
total_red_win.rename({'fighter_red_id': 'fighter_id'}, axis=1, inplace=True)
total_blue_win = target_matches[target_matches['result_id'] == 3][['match_id', 'fighter_blue_id']]
total_blue_win.rename({'fighter_blue_id': 'fighter_id'}, axis=1, inplace=True)
total_win = pd.concat([total_red_win, total_blue_win], ignore_index=True)
total_win['ufc_win'] = 1
fighter_ufc_win = total_win.drop('match_id', axis=1).groupby(['fighter_id'], as_index=False).sum()

## submission 승리 통계. method_id = 0 이 SUB. result_id 0 WL, result_id 3 LW.
## SUB_landed 에서 grappling 이 아닌 것은 제외한다. sub_no_skill 에 해당하고 sub_skill 에 해당하지 않는 서브미션.
sub_no_skill = ['injury', 'fatigue', 'chin to eye', 'verbal', 'position - mount']
sub_skill = ['guillotine', 'choke', 'kimura', 'lock', 'triangle', 'hook', 'triangle', 'omoplata', 'control', 'technical']
sub_no_skill_index = matches[(matches['method_id'] == 0) & (matches['detail'].str.lower().str.contains('|'.join(sub_no_skill))) & (~matches['detail'].str.lower().str.contains('|'.join(sub_skill)))].index
sub_no_skill_index.append(matches[(matches['method_id'] == 0) & (matches['detail'].str.len() == 0)].index)

sub_matches = matches.drop(sub_no_skill_index)

sub_red_win = sub_matches[(sub_matches['method_id'] == 0) & (sub_matches['result_id'] == 0 )][['match_id', 'fighter_red_id']]
sub_red_win.rename({'fighter_red_id': 'fighter_id'}, axis=1, inplace=True)
sub_blue_win = sub_matches[(sub_matches['method_id'] == 0) & (sub_matches['result_id'] == 3 )][['match_id', 'fighter_blue_id']]
sub_blue_win.rename({'fighter_blue_id': 'fighter_id'}, axis=1, inplace=True)
sub_win = pd.concat([sub_red_win, sub_blue_win], ignore_index=True)
sub_win['SUB_landed'] = 1

## KO/TKO 승리 통계. method_id = 1, 2. result_id 0 WL, result_id 3 LW.
ko_red_win = matches[((matches['method_id'] == 1) | (matches['method_id'] == 2)) & (matches['result_id'] == 0 )][['match_id', 'fighter_red_id']]
ko_red_win.rename({'fighter_red_id': 'fighter_id'}, axis=1, inplace=True)
ko_blue_win = matches[((matches['method_id'] == 1) | (matches['method_id'] == 2)) & (matches['result_id'] == 3 )][['match_id', 'fighter_blue_id']]
ko_blue_win.rename({'fighter_blue_id': 'fighter_id'}, axis=1, inplace=True)
ko_win = pd.concat([ko_red_win, ko_blue_win], ignore_index=True)
ko_win['KO'] = 1

## round_sum_by_fighter 에 SUB_landed, KO column 을 추가한다.
round_sum_by_fighter = pd.merge(round_sum_by_fighter, sub_win, how='left', on=['match_id', 'fighter_id'])
round_sum_by_fighter = pd.merge(round_sum_by_fighter, ko_win, how='left', on=['match_id', 'fighter_id'])
round_sum_by_fighter.fillna(value=0, inplace=True)
# round_sum_by_fighter = round_sum_by_fighter[[
#     'match_id', 'fighter_id', 'TD_landed', 'TD_attempted', 'SUB_attempted',
#     'SUB_landed', 'REV', 'CTRL_sec', 'KD', 'KO',
#     'HEAD_landed', 'HEAD_attempted', 'BODY_landed', 'BODY_attempted', 'LEG_landed',
#     'LEG_attempted', 'DISTANCE_landed', 'DISTANCE_attempted', 'CLINCH_landed', 'CLINCH_attempted',
#     'GROUND_landed', 'GROUND_attempted']]

## submission 성공률, KO 갯수 계산
fighter_record_sum = round_sum_by_fighter.drop('match_id', axis=1).groupby(['fighter_id'], as_index=False).sum()
fighter_record_sum['SUB %'] = fighter_record_sum['SUB_landed'] / fighter_record_sum['SUB_attempted']
fighter_record_sum = pd.merge(fighter_record_sum, fighter_ufc_win, on='fighter_id')[['fighter_id', 'SUB %', 'SUB_landed', 'ufc_win']]
fighter_record_sum['SUB_win/win'] = fighter_record_sum['SUB_landed'] / fighter_record_sum['ufc_win']
## 서브미션승 /총승 >=1 이면 1로 수정
fighter_record_sum.loc[fighter_record_sum['SUB_win/win'] > 1, 'SUB_win/win'] = 1
fighter_sub = fighter_record_sum[['fighter_id', 'SUB %', 'SUB_win/win']]
# fighter_ko = fighter_record_sum[['fighter_id', 'KO']]

## match_id, fighter_id 별 경기 기록과  match_id_sec 을 합친다.
match_fighter_sec = pd.merge(round_sum_by_fighter, match_id_sec, on='match_id')
round_div_sec = match_fighter_sec.drop(['match_id', 'fighter_id'], axis=1).div(match_fighter_sec['seconds'], axis='index')
round_div_sec = pd.concat([match_fighter_sec[['match_id', 'fighter_id']], round_div_sec], axis=1)
## 각 match 별 총 round 개수
round_div_sec_round = pd.merge(round_div_sec, round_max_by_fighter, on=['match_id', 'fighter_id'])
round_div_sec_round.drop(['round_number', 'seconds'], axis=1, inplace=True)

## fighter 별로 각 기록의 평균을 낸다. 나중에는 총 경기 횟수도 고려하자.
fighter_record_avg = round_div_sec_round.drop(['match_id'], axis=1).groupby(['fighter_id'], as_index=True).mean()
## submission rate, ko 갯수 을 추가한다.
fighter_record_avg = pd.merge(fighter_record_avg, fighter_sub, on='fighter_id')
## Georges St-Pierre 도 SUB % 가 0.125 이다. 수준 높은 선수랑 싸울수록 성공률이 낮아지기 때문이다.
## SUB % * SUB_WIN/WIN 의 값을 새로운 통계량으로 만들자.
fighter_record_avg['SUB_quotient'] = fighter_record_avg['SUB %'] * fighter_record_avg['SUB_win/win']
# fighter_record_avg = pd.merge(fighter_record_avg, fighter_ko, on='fighter_id')

demo = pd.merge(fighter_record_avg, fighters[['fighter_id', 'fighter_name', 'fighter_nickname']], on='fighter_id')
demo[['fighter_id', 'fighter_name', 'fighter_nickname', 'SUB_quotient']].sort_values(by='SUB_quotient', ascending=False)
demo[demo['fighter_name'].str.contains('Georges')][['fighter_id', 'fighter_name', 'fighter_nickname', 'SUB %', 'SUB_quotient']]
demo[demo['SUB %'] == 1]
len(demo[demo['SUB %'].isna()]) / len(demo)
plt.hist(demo[~demo['SUB_quotient'].isna()]['SUB_quotient'])
plt.hist(demo[~demo['KO'].isna()]['KO'], log=True)
demo[demo['KO'] > 0.01]
demo[demo['fighter_name'].str.contains('|'.join(['Chuck Liddell', 'Rich Franklin', 'Andrei Arlovski', 'Alistair Overeem', 'Dustin Poirier', 'Stipe Miocic', 'Max Holloway', 'Francis Ngannou','Anderson Silva']))]['KO'].sort_values()

ko_par = float(demo[demo['fighter_name'].str.contains('Ngan')]['KO'] / 2)
1 - len(demo[demo['KO'] >= ko_par]) / len(demo)
demo.iloc[demo[demo['KO'] >= demo['KO'].quantile(0.9)]['KO'].sort_values().index, :][['fighter_name', 'KO']]
demo['SUB_quotient'].quantile(0.6, 'nearest')
len(demo[demo['SUB_quotient'] > 0.15]) / len(demo)
# 3. 군집 분석 - grappling
## normalization
# data_g = fighter_record_avg.values
# data_g = fighter_record_avg[['TD_landed', 'TD_attempted', 'SUB_attempted', 'REV', 'CTRL_sec']].values
# data_g = fighter_record_avg[['TD_landed', 'TD_attempted', 'SUB_attempted', 'REV']].values
data_g = fighter_record_avg[['SUB_attempted', 'SUB_landed', 'SUB %']].values
scaler = preprocessing.StandardScaler()
print(scaler.fit(data_g))
# # print(scaler.mean_)
data_normal = scaler.transform(data_g)

## Kmeans clustering
kmeans = KMeans(n_clusters=2).fit(data_normal)
predict = kmeans.predict(data_normal)
centroids = kmeans.cluster_centers_

cluster_g = pd.Series(predict, index=fighter_record_avg.index, name='style_g')
cluster_g = pd.merge(cluster_g, fighters[['fighter_id', 'fighter_name', 'fighter_nickname']], on='fighter_id')

cluster_g[cluster_g['fighter_name'].str.contains('Khabib')]
cluster_g[cluster_g['fighter_name'].str.contains('Covington')]
cluster_g[cluster_g['fighter_name'].str.contains('Dong Hyun Kim')]
cluster_g[cluster_g['fighter_name'].str.contains('Volkanovski')]
cluster_g[cluster_g['fighter_name'].str.contains('Usman')]
cluster_g[cluster_g['fighter_name'].str.contains('Gilbert Burns')]
cluster_g[cluster_g['fighter_name'].str.contains('Demian Maia')]
cluster_g[cluster_g['fighter_name'].str.contains('Ronda')]
cluster_g[cluster_g['fighter_name'].str.contains('Chima')]
cluster_g[cluster_g['fighter_name'].str.contains('Dern')]
cluster_g[cluster_g['fighter_name'].str.contains('Frank Mir')]
cluster_g[cluster_g['fighter_name'].str.contains('Zingano')]
cluster_g[cluster_g['fighter_name'].str.contains('Miesha Tate')]
cluster_g[cluster_g['fighter_name'].str.contains('Werdum')]
cluster_g[cluster_g['fighter_name'].str.contains('Charles Oliveira')]

cluster_g[cluster_g['fighter_name'].str.contains('Miocic')]
cluster_g[cluster_g['fighter_name'].str.contains('Lesnar')]

cluster_g[cluster_g['fighter_name'].str.contains('Valentina Shevchenko')]
cluster_g[cluster_g['fighter_name'].str.contains('Amanda Nunes')]
cluster_g[cluster_g['fighter_name'].str.contains('Adesanya')]
cluster_g[cluster_g['fighter_name'].str.contains('Yoel Romero')]
cluster_g[cluster_g['fighter_name'].str.contains('Conor')]
cluster_g[cluster_g['fighter_name'].str.contains('Dustin Poirier')]
cluster_g[cluster_g['fighter_name'].str.contains('ho Choi')]
cluster_g[cluster_g['fighter_name'].str.contains('Gaethje')]
cluster_g[cluster_g['fighter_name'].str.contains('Ortega')]
cluster_g[cluster_g['fighter_name'].str.contains('Chan Sung')]
cluster_g[cluster_g['fighter_name'].str.contains('Robert Whittaker')]
cluster_g[cluster_g['fighter_name'].str.contains('Megan Anderson')]
cluster_g[cluster_g['fighter_name'].str.contains('Ngannou')]
cluster_g[cluster_g['fighter_name'].str.contains('Holloway')]