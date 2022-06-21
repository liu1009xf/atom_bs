import pandas as pd
import package.core as core


from logging import getLogger
logger = getLogger(__name__)
from tqdm.notebook import tqdm

def analyze(tickets:list[core.RaceTicket], store):
    race_ids = [x.race_id for x in tickets]
    df = pd.DataFrame({'race_id':race_ids})
    race_master = pd.read_hdf(store, key='race_master',table=True, mode='r', where=['race_id = race_ids'])
    df['ticket'] = tickets
    df = pd.merge(df,race_master,on='race_id',how='left')
    df['cost'] = df['ticket'].apply(lambda x: x.bet *100)
    df['return'] = df['ticket'].apply(lambda x: get_return(x, store))
    return df 


def get_ticket_result(ticket: core.RaceTicket, store):
    cond = f'race_id=\"{ticket.race_id}\" & type=\"{ticket.ticket_type_str}\"'
    results=pd.read_hdf(store, key='result',table=True, mode='r', where=[cond])
    if(len(results)==0):
        logger.error("Result not ready")
    return results

def get_return(ticket: core.RaceTicket, store):
    results = get_ticket_result(ticket, store)
    res = results['strike'].iloc[0]
    tc = ticket.content
    if('-' in res):
        res = sorted(res.split('-'))
        tc = sorted(tc.split('-'))
    return results['return'].iloc[0]*ticket.bet if res ==tc else 0
