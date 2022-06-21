import datetime as dt

'''
expecting minutes:seconds:deciseconds.
deciseconds to microseconds: 1:100000

'''
def racetimestr2timedelta(time_str:str) -> dt.time:
    times = time_str.split('.')
    times1 = times[0].split(':')
    time = dt.time(minute= int(times1[0]), second=int(times1[1]),
                        microsecond=int(times[1])*100000)
    return time