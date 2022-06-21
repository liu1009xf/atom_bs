
from dataclasses import dataclass
from enum import Enum
import datetime as dt
import abc
from typing import Callable
import pandas as pd

class TicketType(Enum):
    単勝=1
    複勝=2
    枠連=3
    馬連=4
    ワイド=5
    馬単=6
    三連複=7
    三連単=8

'''
bet should be int with range (1, inf), and the unit of bet is lot.
there is a 100 multiplier for each lot, meaning bet 1 lot means bet 100Yen
'''
class RaceTicket:
    def __init__(self, race_id: str, ticket_type: TicketType, pattern: list[int], bet: int = 1) -> None:
        self.race_id = race_id
        self.ticket_type = ticket_type
        self.pattern = pattern
        self.bet = bet
        self.content=self._build_pattern_string(ticket_type=ticket_type, pattern=pattern)
    
    def __str__(self) -> str:
        return f'Race: {self.race_id}, Type:{self.ticket_type.name}, Patter:{self.content}'
    
    def __repr__(self) -> str:
        return self.__str__()
    
    @property
    def cost(self):
        return self.bet*100

    @property
    def ticket_type_str(self):
        return self.ticket_type.name.replace('三', '3')

    @classmethod
    def _build_pattern_string(cls, ticket_type:TicketType, pattern:list[int]) -> str:
        pattern_str = [str(x) for x in pattern]
        if(ticket_type in [TicketType.枠連, TicketType.馬連, TicketType.ワイド, TicketType.三連複]):
            return cls._build_dash_pattern(pattern_str)
        elif(ticket_type in [TicketType.複勝 or TicketType.単勝]):
            return pattern_str[0]
        else:
            return cls._build_arrow_pattern(pattern_str)

    @classmethod
    def _build_dash_pattern(cls, pattern):
        return '-'.join(pattern)

    @classmethod
    def _build_arrow_pattern(cls, pattern):
        return '→'.join(pattern)

@dataclass
class SchedEvent:
    def __init__(self, time:dt.datetime, callback:Callable[[dict], None], is_before_race = True) -> None:
        self.time = time
        self.callback = callback
        self.is_before_race = is_before_race

class Service(abc.ABC):
    @abc.abstractmethod
    def buy_ticket(self, ticket:RaceTicket) -> None: 
        raise NotImplementedError("buy ticket must be implemented")
    
    @abc.abstractmethod
    def register_sched_event(self, sched_event: SchedEvent):
        raise NotImplementedError("register_sched_event must be implemented")

    @abc.abstractmethod
    def get_next_race(self):
        raise NotImplementedError("get_next_race must be implemented")



class Paper(Service):
    def __init__(self, 
                store:pd.HDFStore,  
                *,
                start:dt.datetime, 
                end:dt.datetime, 
                cash:int = 10000,
                ) -> None:
        super().__init__()
        self.cash = cash
        self.holding_tickets = list()
        self.events = list()
        self.start = start
        self.end = end

        # self.__races = pd.read_hdf(store, key='race',table=True, mode='r', where=['date>=start & date<=end'])
        # self.__result = pd.read_hdf(store, key='result',table=True, mode='r', where=['date>=start & date<=end'])
        # self.__race_master = pd.read_hdf(store, key='race_master',table=True, mode='r', where=['date>=start & date<=end'])
        self.__races = pd.read_hdf(store, key='race',table=True, mode='r')
        self.__result = pd.read_hdf(store, key='result',table=True, mode='r')
        self.__race_master = pd.read_hdf(store, key='race_master',table=True, mode='r')

        self.__races.sort_values(by=["date","start_time"])
        self.next_race_id = None

    @property
    def hist_races(self):
        return self.__races.loc[self.__races['race_id'] == self.next_race_id] if self.next_race_id else None

    def next_race_time(self):
        return 
        
    def get_next_race(self):
        return 

    def buy_ticket(self, ticket: RaceTicket) -> bool:
        amount =  ticket.bet*100
        if amount < self.cash:
            print(f"not enough cash to buy {str(ticket)}") 
        self.cash -= ticket.bet*100
        return True
    
    def register_sched_event(self, sched_event:SchedEvent):
        self.events.append(sched_event)
    


class Strategy(abc.ABC):

    def _set_hist(self, hist):
        self.hist = hist
    
    def _next_data(self, data):
        self.next_data = data
    
    def _set_service(self, service):
        self.service = service

    @abc.abstractmethod
    def sched(self) -> list[SchedEvent]:
        raise NotImplementedError()
    
    @abc.abstractmethod
    def next_race(self) -> None:
        raise NotImplementedError()

class Engine:

    def __init__(self, 
                service:Service, 
                strategy:Strategy):
        self.service = service
        self.ticket_bought = list()
        self.strategy = strategy
        for event in strategy.sched():
            self.service.register_sched_event(sched_event=event)
        
    def run(self) -> None:
        data = self.service.get_next_race()
        self.strategy._next_data(data)
        self.strategy.next_race()
        self.ticket_bought.append(RaceTicket)
        
