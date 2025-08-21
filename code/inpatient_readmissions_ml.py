import pyodbc
import pandas as pd
import numpy as np
from sqlalchemy import create_engine

conn_str = (
    r'DRIVER={SQL Server};'
    r'SERVER=lct-sqlbidev\dev;'
    r'DATABASE=Informatics_SSAS_Live;'
    r'Trusted_Connection=yes;'
    )

cnxn = pyodbc.connect(conn_str) # connect using the connection string

cursor_contacts = cnxn.cursor()

cursor_contacts.execute("EXEC [Informatics_SSAS_Live].[Reporting]."
               "[usp_HSMAProject_DNAs_process]") # the sql we want to run

readmissions_data = cursor_contacts.fetchall() # return all the data

# get list of headers using list comprehension - this will account for new 
# columns dynamically as they are added to the SQL source data
contact_headers = [column[0] for column in cursor_contacts.description] 

#headers

# load data into pandas dataframe
readmissions_df = pd.DataFrame(np.array(readmissions_data),
                                columns = contact_headers)

readmissions_df['ReAdmission'] = readmissions_df['ReAdmission'].astype(int)