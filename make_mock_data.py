"""Generate tiny fake CSVs with the Home Credit schema, to smoke-test the pipeline offline.

The numbers are meaningless: the point is that every table, key and column name the
pipeline touches exists, so a run exercises the full code path in seconds. Never use the
output for any reported result.
"""
# Usage: python make_mock_data.py ./mock_data && python home_credit_pipeline.py --data ./mock_data --out ./mock_res
import numpy as np, pandas as pd, os
rng=np.random.default_rng(0); import sys; d=sys.argv[1] if len(sys.argv)>1 else "./mock_data"; os.makedirs(d,exist_ok=True)
def app(n,start,target):
    df=pd.DataFrame({"SK_ID_CURR":np.arange(start,start+n),
      "NAME_CONTRACT_TYPE":rng.choice(["Cash loans","Revolving loans"],n),
      "CODE_GENDER":rng.choice(["M","F","XNA"],n,p=[.45,.54,.01]),
      "AMT_INCOME_TOTAL":rng.lognormal(12,.5,n),"AMT_CREDIT":rng.lognormal(13,.6,n),
      "AMT_ANNUITY":rng.lognormal(10,.5,n),"AMT_GOODS_PRICE":rng.lognormal(13,.6,n),
      "DAYS_BIRTH":-rng.integers(7000,25000,n),"DAYS_EMPLOYED":np.where(rng.random(n)<.18,365243,-rng.integers(0,15000,n)),
      "CNT_FAM_MEMBERS":rng.integers(1,6,n).astype(float),
      "EXT_SOURCE_1":np.where(rng.random(n)<.56,np.nan,rng.random(n)),"EXT_SOURCE_2":rng.random(n),
      "EXT_SOURCE_3":np.where(rng.random(n)<.2,np.nan,rng.random(n)),"OWN_CAR_AGE":np.where(rng.random(n)<.66,np.nan,rng.integers(0,30,n))})
    if target:
        z=-2.5-2*df.EXT_SOURCE_2+1.0*df.EXT_SOURCE_1.isna()
        df.insert(1,"TARGET",(rng.random(n)<1/(1+np.exp(-z))).astype(int))
    return df
tr=app(3000,100000,True); te=app(500,200000,False); tr.to_csv(f"{d}/application_train.csv",index=False); te.to_csv(f"{d}/application_test.csv",index=False)
ids=np.r_[tr.SK_ID_CURR,te.SK_ID_CURR]; has=rng.choice(ids,int(.85*len(ids)),replace=False)
b=pd.DataFrame({"SK_ID_CURR":rng.choice(has,6000),"SK_ID_BUREAU":np.arange(6000),"CREDIT_ACTIVE":rng.choice(["Active","Closed"],6000),
 "CREDIT_TYPE":rng.choice(["Consumer credit","Credit card"],6000),"DAYS_CREDIT":-rng.integers(0,3000,6000),
 "AMT_CREDIT_SUM":rng.lognormal(12,1,6000),"AMT_CREDIT_SUM_DEBT":rng.lognormal(10,1,6000)}); b.to_csv(f"{d}/bureau.csv",index=False)
pd.DataFrame({"SK_ID_BUREAU":rng.integers(0,6000,20000),"MONTHS_BALANCE":-rng.integers(0,60,20000),"STATUS":rng.choice(list("0012CX"),20000)}).to_csv(f"{d}/bureau_balance.csv",index=False)
p=pd.DataFrame({"SK_ID_PREV":np.arange(8000),"SK_ID_CURR":rng.choice(ids,8000),"AMT_APPLICATION":rng.lognormal(12,1,8000),"AMT_CREDIT":rng.lognormal(12,1,8000),
 "NAME_CONTRACT_STATUS":rng.choice(["Approved","Refused","Canceled"],8000),"NAME_CONTRACT_TYPE":rng.choice(["Cash loans","Consumer loans"],8000)})
for c in ["DAYS_FIRST_DRAWING","DAYS_FIRST_DUE","DAYS_LAST_DUE_1ST_VERSION","DAYS_LAST_DUE","DAYS_TERMINATION"]: p[c]=np.where(rng.random(8000)<.3,365243,-rng.integers(0,3000,8000))
p.to_csv(f"{d}/previous_application.csv",index=False)
pd.DataFrame({"SK_ID_PREV":rng.integers(0,8000,20000),"SK_ID_CURR":rng.choice(ids,20000),"MONTHS_BALANCE":-rng.integers(0,90,20000),"SK_DPD":rng.poisson(.3,20000)}).to_csv(f"{d}/POS_CASH_balance.csv",index=False)
pd.DataFrame({"SK_ID_PREV":rng.integers(0,8000,30000),"SK_ID_CURR":rng.choice(ids,30000),"AMT_INSTALMENT":rng.lognormal(9,1,30000),"AMT_PAYMENT":rng.lognormal(9,1,30000),
 "DAYS_INSTALMENT":-rng.integers(0,3000,30000),"DAYS_ENTRY_PAYMENT":-rng.integers(0,3000,30000)}).to_csv(f"{d}/installments_payments.csv",index=False)
pd.DataFrame({"SK_ID_PREV":rng.integers(0,8000,10000),"SK_ID_CURR":rng.choice(ids,10000),"AMT_BALANCE":rng.lognormal(10,1,10000),"AMT_CREDIT_LIMIT_ACTUAL":np.where(rng.random(10000)<.05,0,rng.lognormal(11,1,10000))}).to_csv(f"{d}/credit_card_balance.csv",index=False)
