
from enum import Enum
import datetime as dt
import abc
from typing import Callable, Optional

class TicketType(Enum):
    単勝=1
    複勝=2
    枠連=3
    馬連=4
    ワイド=5
    馬単=6
    三連複=7
    三連単=8

class RaceTicket:
    def __init__(self, race_id: str, ticket_type: TicketType, pattern: list[int], bet: int = 100) -> None:
        self.race_id = race_id
        self.ticket_type = ticket_type
        self.pattern = pattern
        self.bet = bet
        self.content=self._build_pattern_string(ticket_type=ticket_type, pattern=pattern)
    
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

class SchedEvent:
    def __init__(self, time:dt.datetime, callback:Callable[[dict], None]) -> None:
        self.time = time
        self.callback = callback

class Service(abc.ABC):
    @abc.abstractmethod
    def buy_ticket(self, ticket:RaceTicket) -> None: 
        raise NotImplementedError("buy ticket must be implemented")

    @abc.abstractmethod
    def get_result(self, ticket:RaceTicket) -> dict:
        raise NotImplementedError("get_result must be implemented")
    
    '''
    Season end is defined as the first monday after each weekend
    '''
    @abc.abstractmethod
    def is_season_ends(self) -> bool:
        raise NotImplementedError("get_result must be implemented")

    @abc.abstractmethod
    def register_sched_event(self, sched_event: SchedEvent):
        raise NotImplementedError("register_sched_event must be implemented")

    @abc.abstractmethod
    def get_next_race(self):
        raise NotImplementedError("get_next_race must be implemented")


class Paper(Service):
    def __init__(self, 
                cash:int = 10000, 
                start=dt.datetime, 
                end=dt.datetime, 
                season_end_cra:Optional[Callable[[dt.datetime, dt.datetime], bool]] = None
                ) -> None:
        super().__init__()
        self.cash = cash
        self.holding_tickets = list()
        self.events = list()
        self.start = start
        self.end = end
        self.curr_race_date = start
        self.next_race_date = start
        
        self.season_end_cra= season_end_cra or self._season_end_cra
        
    def buy_ticket(self, ticket: RaceTicket) -> bool:
        self.cash -= ticket.bet
        return True
    
    def register_sched_event(self, sched_event:SchedEvent):
        self.events.append(sched_event)

    
    @classmethod
    def _season_end_cra(cls, cur:dt.datetime, next:dt.datetime):
        tmr= cur+dt.timedelta(days=1)
        return next> tmr
    
    def is_season_ends(self) -> bool:
        return self.season_end_cra(self.curr_race_date, self.next_race_date)



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
        