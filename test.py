from model.model import Recommendation_MODEL
import datetime
import time

dt = datetime.datetime.strptime(f'03/16/23 11:00', '%m/%d/%y %H:%M')
millisenconds = int(time.mktime(dt.timetuple())) * 1000 # input from web
print(millisenconds)

# trip_level = {"art":5, "history":1, "nature":1, "shopping":5}
# model = Recommendation_MODEL(trip_level= list(trip_level.values()))

# planning = model.planning(milli_start_time=millisenconds, placePerDay=3, day=3, timePerDay=9)
# print(planning)

# import datetime
# import time

# dt = datetime.datetime.strptime('03/11/23 18:00', '%m/%d/%y %H:%M')

# millisenconds = int(time.mktime(dt.timetuple())) * 1000
# print(millisenconds)