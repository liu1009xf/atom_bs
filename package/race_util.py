import datetime as dt

'''
expecting minutes:seconds:deciseconds.
deciseconds to microseconds: 1:100000

'''
def racetimestr2timedelta(time_str:str) -> dt.timedelta:
    times = time_str.split('.')
    times1 = times[0].split(':')
    time = dt.timedelta(seconds=int(times1[0])*60+int(times1[1]),
                        microseconds=int(times[1])*100000)
    return time