# -*- coding: utf-8 -*-
"""
-------------------------------------------------------------------------------
Created: Feb 13 2018
Updated: Feb 15, 2018
Author: Heather CLifford
Email: heather.clifford@maine.edu

LA-ICP-MS Software

Use: Creates excel and pdf documents containing compiled and edited data from
     information provided by the user and raw data provided by the output of
     the LA-ICP-MS system.
-------------------------------------------------------------------------------
"""
import sys
import os
import re
import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy.interpolate import interp1d
from pandas import DataFrame,Series
from typing import Tuple,List

def read_input_information(input_files:str)-> DataFrame:
    """
    Reads input information from InputInformation sheet in the input file excel
        documents found in Input_Files folder to a compiled dataframe

    Input:
        input_files: Input_Files directory containing input files

    Output:
        df: Dataframe containing information from the InputInformation sheet
            from the input file excel documents found in Input_Files folder
    """

    input_information = [pd.read_excel(os.path.join(input_files,i),
                        sheetname='InputInformation',index_col=0,header=[0,1,2],
                        skiprows=2).T for i in os.listdir(input_files) if i.startswith(('Input','input'))]
    df = pd.concat(input_information)
    df['Date Ran'] =  pd.to_datetime(df['Date Ran'], format="%m/%d/%Y")
    print('Running LAICPMS Software for {} Core'.format(df.index.get_level_values(0)[0]))

    return df

def export_tuning_parameters(input_files:str,input_information:DataFrame,
        laser_directory:str,abbreviation:str)->DataFrame:
    """
    Reads tuning parameter information from InputInformation sheet in the input
        file excel documents found in Input_Files folder to a compiled dataframe

    Input:
        input_files: Input_Files directory containing input files
        input_information: Dataframe containing compiled input information
        laser_directory: directory from user input
        abbreviation: core abbreviation from input information

    Output:
        df: Dataframe containing information from the TuningParameters sheet
            from the input file excel documents found in Input_Files folder
    """

    now = datetime.datetime.now()
    folder = os.path.join(laser_directory, 'Output_Data_Files')
    if not os.path.exists(folder):
        os.makedirs(folder)
    tuning_parameters= [pd.read_excel(os.path.join(input_files,i),sheetname='TuningParameters',header=None,index_col=0,
                                       skiprows=2).T for i in os.listdir(input_files) if i.startswith(('Input','input'))]

    df = pd.concat(tuning_parameters)
    df = df.set_index(input_information.index)
    df.insert(0,'Date Ran',input_information['Date Ran'])
    df.insert(1,'Name',input_information['Name of Person'])
    excel_name = pd.ExcelWriter(os.path.join(folder,'{}_TuningParameters_{}.xlsx'.format
                                             (abbreviation,now.strftime("%m%d%Y"))),datetime_format= 'mm/dd/yy')
    df.to_excel(excel_name)
    excel_name.save()
    return df

def read_run_logs(input_files:str,input_information:DataFrame)->Tuple[dict,str]:


    runlogs = [pd.read_excel(os.path.join(input_files,i),sheetname='RunLog',index_col=[0,1],header=0,
                                       skiprows=2) for i in os.listdir(input_files) if i.startswith(('Input','input'))]
    if len(input_information.index.levels[1].tolist())>1:
        print('more than one core abbreviation found,only can process\
                  one core at a time')
    else:
        abbreviation=input_information.index.levels[1].tolist()[0]

    sections=[abbreviation + str(i) for i in input_information.index.levels[2].tolist()]

    section_runs={}
    for i in range(len(sections)):
        section_runs[sections[i]]=runlogs[i]

    return section_runs,abbreviation

def export_data(sections:str,section_runs:dict,laser_directory:str,abbreviation:str)->Tuple[DataFrame,DataFrame]:
    background_data1 = {}
    background_data2 = {}

    data1=[]
    data2=[]
    top=[]
    bot = []
    for i in os.listdir(sections):
        print('Processing Section {}'.format(i))
        section_data1=[]
        section_data2=[]
        indexes=[]
        bck1={}
        bck2={}

        for index, row in section_runs[i].iterrows():

            indexes.append(index)

            file1,file2=file_names(sections,i,index)

            data1.append(run(file1,row))
            data2.append(run(file2,row))

            section_data1.append(run(file1,row))
            section_data2.append(run(file2,row))

            bck1[index]=run_background(file1)
            bck2[index]=run_background(file2)

        export_section_plot(pd.concat(section_data1),pd.concat(section_data2),i,
                            laser_directory,top_bot_depths(i,section_runs))
        export_run_plot(section_data1,section_data2,i,laser_directory,indexes)
        depths = top_bot_depths(i,section_runs)
        top.extend(depths[0])
        bot.extend(depths[1])
        background_data1[i]=pd.concat(bck1.values(), keys=bck1.keys())
        background_data2[i]=pd.concat(bck2.values(), keys=bck2.keys())


    export_background_stats(pd.concat(background_data1.values(), keys=background_data1.keys()),'MR',laser_directory)
    export_background_stats(pd.concat(background_data2.values(), keys=background_data2.keys()),'LR',laser_directory)

    export_core_plot(pd.concat(data1),pd.concat(data2),abbreviation,laser_directory,top,bot)


    return pd.concat(data1),pd.concat(data2)

def load_txt_file(file_path:str)-> DataFrame:

    rows = []
    if not (os.path.splitext(file_path)[1]=='.txt') | (os.path.splitext(file_path)[1]=='.TXT'):
        file_path = file_path + '.txt'

    with open(file_path) as f:
        for i, line in enumerate(f):
            if i == 0:
                depth_header = line.split("\t")
                if len(depth_header) ==1:
                    depth_header = line.split(",")

                depth_header[0] = "Time"
                for i in range(1, len(depth_header)):
                    depth_header[i] = re.sub("\(.*\)", "", depth_header[i]).strip()
            elif i > 5:
                row=line.split()
                if len(row) ==1:
                    row=line.split(',')
                    row=row[:-1]
                rows.append(row)
    return pd.DataFrame(rows, columns=depth_header)

def add_depth(run:DataFrame,row:DataFrame)->DataFrame:
    top = row['Top Depth (cm)']/100
    bot = row['Bottom Depth (cm)'] /100
    inc = (bot- top) / (run.shape[0])
    depth_series = pd.Series(np.arange(top, bot, inc))
    run.insert(0, 'Depth (m)', depth_series)
    return run.set_index('Depth (m)')

def run(file:str,row:DataFrame)->DataFrame:
    run = load_txt_file(file)
    run = run.astype(float)
    run = run.apply(np.round)
    run = run[(run['Time']>12) & (run['Time']<(row['Laser Time (s)']-23))]
    run = run.reset_index(drop=True)
    run = run.drop('Time', 1)
    run = add_depth(run,row)
    return run

def export_section_plot(df1:DataFrame,df2:DataFrame,section:str,
                        laser_directory:str,depths:List[float]):
    folder = os.path.join(laser_directory, 'Output_Data_Files','Master_Plots')
    if not os.path.exists(folder):
        os.makedirs(folder)

    folder2 = os.path.join(folder,section)
    if not os.path.exists(folder2):
        os.makedirs(folder2)

    with PdfPages(os.path.join(folder2,'{}.pdf'.format(section))) as pdf:
        for i in df1.columns:
            plot_section(df1[i],i,pdf,depths[0],depths[1])
        for i in df2.columns:
            plot_section(df2[i],i,pdf,depths[0],depths[1])

def export_core_plot(df1:DataFrame,df2:DataFrame,abbreviation:str,
                        laser_directory:str,top:List[float],bot:List[float]):
    folder = os.path.join(laser_directory, 'Output_Data_Files','Master_Plots')
    if not os.path.exists(folder):
        os.makedirs(folder)

    with PdfPages(os.path.join(folder,'{}_Full_Core.pdf'.format(abbreviation))) as pdf:
        for i in df1.columns:
            plot_core(df1[i],i,pdf,top,bot)
        for i in df2.columns:
            plot_core(df2[i],i,pdf,top,bot)

def plot_section(s:Series,sample:str,pdf,top:List[float],bot:List[float]):

    ax = s.plot(figsize=(8,4),logy=True)
    ax.legend(['{}_raw'.format(sample)])

    for t in top:
        ax.axvline(x=t, linestyle = '--',color = 'lightgrey')

    for b in bot:
        ax.axvline(x=b, linestyle = '--',color = 'burlywood')

    plt.tight_layout()
    fig = plt.gcf()
    pdf.savefig(fig)
    plt.close()

def plot_core(s:Series,sample:str,pdf,top:List[float],bot:List[float]):

    ax = s.plot(figsize=(8,4),logy=True,color='gray',legend=True)
    smooth = cubic_spline(s,sample,20,'cubic')
    smooth.plot(ax=ax,color='red')
    ax.legend(['{}_raw'.format(sample),'{}_cubicspline'.format(sample)])


    for t in top:
        ax.axvline(x=t, linestyle = '--',color = 'lightgrey')

    for b in bot:
        ax.axvline(x=b, linestyle = '--',color = 'burlywood')

    plt.tight_layout()
    fig = plt.gcf()
    pdf.savefig(fig)
    plt.close()

def plot_run(df:DataFrame,pdf,index:List[float],section:str):

    num_samples = len(df.columns)
    figure,ax = plt.subplots(num_samples,figsize=(8,10),sharex=True)
    ax[0].set_title('{} Runs {},{}'.format(section,index[0],index[1]))
    for i,col in enumerate(df.columns):
        df[col].plot(ax=ax[i],color='gray',legend=True,logy=True)
        smooth = cubic_spline(df[col],col)
        smooth.plot(ax=ax[i],color='red')
        ax[i].legend(['{}_raw'.format(col),'{}_cubicspline'.format(col)])

    plt.tight_layout()
    plt.subplots_adjust(hspace=.0)
    fig = plt.gcf()
    pdf.savefig(fig)
    plt.close()

def cubic_spline(s:Series,sample:str,num:int=6,kind:str='cubic')->Series:

    interp = interp1d(s.index,s.values,kind='cubic')
    xnew = np.linspace(s.index.min(),s.index.max(),len(s)/num)
    ynew=interp(xnew)

    return pd.Series(ynew,index=xnew)

def export_run_plot(list1:List[DataFrame],list2:List[DataFrame],section:str,
    laser_directory:str,indexes:List[float]):
    folder = os.path.join(laser_directory, 'Output_Data_Files','Master_Plots')
    if not os.path.exists(folder):
        os.makedirs(folder)

    folder2 = os.path.join(folder,section)
    if not os.path.exists(folder2):
        os.makedirs(folder2)

    with PdfPages(os.path.join(folder2,'{}_Runs.pdf'.format(section))) as pdf:
        for i,run in enumerate(list1):
            plot_run(list1[i],pdf,indexes[i],section)
            plot_run(list2[i],pdf,indexes[i],section)

def run_background(file:str)->DataFrame:
    run = load_txt_file(file)
    run = run.astype(float)
    run = run.apply(np.round)
    run = run.set_index('Time')
    background_stats = run[run.index<12].describe()

    return background_stats

def export_background_stats(df:DataFrame,resolution:str,laser_directory:str):
    folder = os.path.join(laser_directory, 'Output_Data_Files')
    if not os.path.exists(folder):
        os.makedirs(folder)
    folder2 = os.path.join(folder, 'Background_Information')
    if not os.path.exists(folder2):
        os.makedirs(folder2)
    writer = pd.ExcelWriter(os.path.join(folder2,'Run_Background_Stats_{}.xlsx'.format(resolution)))
    with PdfPages(os.path.join(folder2,'Run_Background_Stats_{}.pdf'.format(resolution))) as pdf:
        for i,stat in df.groupby(level=3,axis=0,as_index=False):
            stat.plot(subplots=True,figsize = (8,10),drawstyle = 'steps-mid',title='Statistic: {}'.format(i))
            plt.tight_layout()
            plt.subplots_adjust(hspace=.01)
            fig = plt.gcf()
            pdf.savefig(fig)
            plt.close()
            stat.to_excel(writer,sheet_name=i,index_label=['Section','RunMR','RunLR','Stat'])
    writer.save()

def top_bot_depths(section:str,section_runs:dict)->List[List[float]]:
    top = []
    bot = []
    for index, row in section_runs[section].iterrows():
        top.append(row['Top Depth (cm)']/100)
        bot.append(row['Bottom Depth (cm)'] /100)
    return [top,bot]

def file_names(sections:str,section:str,index:List[float])->Tuple[str,str]:

    try:
        file1 = os.path.join(sections,section,str(index[0])+'.txt')
    except:
        file1 = os.path.join(sections,section,str(index[0])+'.TXT')

    try:
        file2 = os.path.join(sections,section,str(index[1])+'.txt')
    except:
        file2 = os.path.join(sections,section,str(index[1])+'.TXT')
    return file1,file2

def laser_directories(laser_directory:str)->Tuple[str,str]:
    laser_directory_folders = os.listdir(laser_directory)
    for i,val in enumerate(laser_directory_folders):
        if val=='Sections':
            sections = os.path.join(laser_directory,laser_directory_folders[i])
        elif val=='Input_Files':
            input_files = os.path.join(laser_directory,laser_directory_folders[i])
    return sections,input_files

def section_run_information(sections:str,section_runs:dict,abbreviation:str)->DataFrame:
    section_info={}
    for i in os.listdir(sections):
        run_info={}

        for index, row in section_runs[i].iterrows():

            run_info['{},{}'.format(index[0],index[1])]=row.to_frame().T

        section_info[i]=pd.concat(run_info.values(), keys=run_info.keys())

    df = pd.concat(section_info.values(), keys=section_info.keys())
    df.index=df.index.droplevel(level=1)

    return df

def export_output(laser_directory:str,abbreviation:str,input_information:DataFrame,
                  sr_info:DataFrame,dataLR:DataFrame,dataMR:DataFrame):
    now = datetime.datetime.now()
    folder = os.path.join(laser_directory, 'Output_Data_Files')
    if not os.path.exists(folder):
        os.makedirs(folder)

    excel_name = pd.ExcelWriter(os.path.join(folder,'{}_Core_LAICPMS_Data_{}.xlsx'.format
                                             (abbreviation,now.strftime("%m%d%Y"))),datetime_format= 'mm/dd/yy')
    input_information.to_excel(excel_name,sheet_name='Input_Information')
    sr_info.to_excel(excel_name,sheet_name='Section_Run_Information',index_label=['Section','Runs_MR','Runs_LR'])
    dataMR.to_excel(excel_name,sheet_name='LAICPMS_Data_Res_MR')
    dataLR.to_excel(excel_name,sheet_name='LAICPMS_Data_Res_LR')

    excel_name.save()

laser_directory = os.path.join(os.getcwd(),sys.argv[1])
sections,input_files = laser_directories(laser_directory)
input_information = read_input_information(input_files)

section_runs,abbreviation = read_run_logs(input_files,input_information)
tuning_parameters = export_tuning_parameters(input_files,input_information,laser_directory,abbreviation)
sr_info = section_run_information(sections,section_runs,abbreviation)
dataMR,dataLR = export_data(sections,section_runs,laser_directory,abbreviation)
export_output(laser_directory,abbreviation,input_information,sr_info,dataLR,dataMR)
