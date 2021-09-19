#!/usr/bin/env python
# coding: utf-8

def splitter(data):
    a,b,c = data.split('/')
    return int(a), int(b), int(c)

def days_calculater(data):
    s_calender={0:0,1:31,2:28,3:31,4:30,5:31,6:30,7:31,8:31,9:30,10:31,11:30,12:31}
    D,M,Y = splitter(data)
    number_leap_year=0
    leap_year=[]
    for x in range(1900,Y+1):
        if x % 4 == 0 and x %100 !=0:
            number_leap_year+=1
            leap_year.append(x)
        elif x % 4 == 0 and x % 100 == 0 and x % 400 == 0:
            number_leap_year+=1
            leap_year.append(x)
    if Y <1904:
        days = 365 * (Y-1900-1)
        days2 = sum([s_calender[x] for x in range(1,M)])
        total_days = days +days2 +D
    elif Y != leap_year[-1]:
        days = 365 * (Y-1900-1) + len(leap_year)
        days2 = sum([s_calender[x] for x in range(1,M)])
        total_days = days + days2 + D
    elif Y == leap_year[-1] and M >2:
        days = 365 * (Y-1900-1) + (len(leap_year)-1)
        days2 = sum([s_calender[x] for x in range(1,M)])+1
        total_days = days + days2 + D
    elif Y == leap_year[-1] and M <=2:
        days = 365 * (Y-1900-1) + (len(leap_year)-1)
        days2 = sum([s_calender[x] for x in range(1,M)])
        total_days = days + days2 + D

    return total_days



while True:
    date1 = input('Please enter day 1:')
    date2 = input('please enter day 2:')
    output = abs(days_calculater(date1) - days_calculater(date2))-1
    print('{} days'.format(output))
