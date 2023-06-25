import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import fnmatch
import math

from scipy.stats import pearsonr

# eeg segment
def eeg_segment(eeg_data, fs, seg_length, eeg_start_time):
    seg_points = seg_length * fs
    skip_second = (60 - int(eeg_start_time[-2:])) % 60
    print('skip_second', skip_second)
    eeg_data = eeg_data[:, skip_second*fs:]
    seg_num = eeg_data.shape[1] // seg_points
    eeg_data = eeg_data[:, 0:seg_points*seg_num]
    eeg_data_seg = eeg_data.reshape((eeg_data.shape[0], -1, seg_points))
    eeg_data_seg = eeg_data_seg.transpose(1,0,2)
    print('eeg_data_seg.shape', eeg_data_seg.shape)
    return eeg_data_seg

def draw_scale(x, draw_max, draw_min):
    if x.ndim == 2:
        x_draw = np.mean(x, axis=1)
    elif x.ndim == 1:
        x_draw = x.copy()
    tmp_scale = (draw_max-draw_min) / (np.max(x_draw) - np.min(x_draw))
    x_draw *= tmp_scale
    x_draw -= (np.max(x_draw) - draw_max)
    return x_draw

# get exp time
def get_exp(path_root, exp_date):
    exp = os.listdir(path_root)
    if '.DS_Store' in exp:
        exp.remove('.DS_Store')
    if exp_date=='0428': # tmp (no xls)
        exp.remove('102044')
    elif exp_date=='0516':
        exp.remove('113549')
        exp.remove('151256')
        exp.remove('155401')
        exp.remove('163619')
        exp.remove('174140')
    elif exp_date=='0517':
        exp.remove('151039')
        exp.remove('155225')
    elif exp_date=='0519':
        exp.remove('082013')
        exp.remove('084443')
        exp.remove('093127')
    elif exp_date=='0523':
        exp.remove('2023-05-23_083431')
        exp.remove('2023-05-23_125103')
    elif exp_date=='0524':
        exp.remove('2023-05-24_082526')
    exp.sort()
    return exp

# get eeg start time and end time
def get_eeg_time(path_root, exp):
    eeg_start_time = []
    eeg_end_time = []
    for i in range(len(exp)):
        with open('{}/{}/info.txt'.format(path_root, exp[i]), 'r') as f:
            for line in f.readlines():
                line = line.strip('\n')
                if '开始时间' in line:
                    eeg_start_time.append(line[-20:-1])
                    break
        with open('{}/{}/info.txt'.format(path_root, exp[i]), 'r') as f:
            for line in f.readlines():
                line = line.strip('\n')
                if '结束时间' in line:
                    eeg_end_time.append(line[-20:-1])
                    break
    return eeg_start_time, eeg_end_time

def csv2eeg(path_root, exp):
    path_tmp = '{}/{}/EP'.format(path_root, exp)
    print('path_tmp', path_tmp)
    if os.path.exists(path_tmp):
        csv_list = []
        for root, _, filenames in os.walk(path_tmp):
            for filename in fnmatch.filter(filenames, '*.csv'):
                csv_list.append(os.path.join(root, filename))
        csv_list.sort()
        print('csv_list', csv_list)
    else:
        return 0

    # read data of NeuraMatrix
    eeg = []
    for c in range(len(csv_list)):
        df = pd.read_csv('{}'.format(csv_list[c]), header=None)
        eeg.append(df[2])
    eeg = np.array(eeg)

    # 有可能ref电极没有贴好，这种情况下会耦合大量的工频干扰
    # eeg -= eeg[0,:]
    # eeg -= np.mean(eeg[[0,1,3,4], :], axis=0)

    return eeg

def draw_eeg_psd(raw, ch_names, exp):
    # fig, ax = plt.subplots(figsize=(6, 6))
    # raw.plot(duration=50, remove_dc=False, scalings='auto')
    raw.plot(duration=50, remove_dc=False, scalings=400)
    
    # fig, ax = plt.subplots(figsize=(8, 4))
    # raw.plot_psd(fmax=499, average=True, ax=ax)
    raw.compute_psd(tmax=np.inf, fmax=499).plot()
    # fig.savefig('./figs/4.28/exp{}_psd.png'.format(exp), dpi=300)
    # for c in ch_names:
    #     fig, ax = plt.subplots(figsize=(8, 4))
    #     raw.plot_psd(fmax=499, average=False, picks=[c], ax=ax)
        # fig.savefig('./figs/4.24/exp{}_psd_ch{}.png'.format(i, c), dpi=300)

def process_xls(csv_path, exp_id, eeg_start_time, eeg_end_time):
    xls_info = {}
    csv_list = os.listdir(csv_path)
    if '.DS_Store' in csv_list:
        csv_list.remove('.DS_Store')
    csv_list.sort()
    print('csv_list', csv_list)

    df = pd.read_excel('{}/{}'.format(csv_path, csv_list[exp_id]))
    print(df)

    hr_line = -1
    bp_line = -1
    bis_line = -1
    event_line = -1
    for i in range(df.iloc[:,0].shape[0]):
        if df.iloc[i,0] == '心率':
            hr_line = i
        elif df.iloc[i,0] == '无创收缩压':
            bp_line = i
        elif df.iloc[i,0] == 'BIS':
            bis_line = i
        elif df.iloc[i,0] == 'EVENT':
            event_line = i

    # event_line = -1 # TODO: 确定event的标记规则
    # print('[debug] df.iloc[event_line, :]', df.iloc[event_line, :])

    # 实际上在xls开始时的时间间隔可能不是1min，但这一段我们往往不关心，认为其间隔是1min也关系不大
    xls_info['time_num'] = len(df.keys()) - 1
    print('time_num', xls_info['time_num'])

    xls_info['time_inject'] = -1
    xls_info['time_loc'] = -1
    xls_info['time_wake'] = -1
    xls_info['time_roc'] = -1
    xls_info['time_druge'] = -1
    xls_info['time_move'] = []
    for i in range(1, xls_info['time_num']):
        if '诱导给药' in str(df.iloc[event_line, i]):
            xls_info['time_inject'] = i
        if '睫毛反射消失' in str(df.iloc[event_line, i]):
            xls_info['time_loc'] = i
        if '唤醒' in str(df.iloc[event_line, i]):
            xls_info['time_wake'] = i
        if '睫毛反射恢复' in str(df.iloc[event_line, i]):
            xls_info['time_roc'] = i
        if '0428' in csv_path:
            str_druge = '停止操作'
        else:
            str_druge = '停药'
        if str_druge in str(df.iloc[event_line, i]):
            xls_info['time_druge'] = i
        if '动' in str(df.iloc[event_line, i]):
            xls_info['time_move'].append(i)
    print('time_inject', xls_info['time_inject'])
    print('time_loc', xls_info['time_loc'])
    print('time_wake', xls_info['time_wake'])
    print('time_roc', xls_info['time_roc'])

    xls_info['hr_start'] = 1
    for i in range(1, xls_info['time_num']+1):
        if not (pd.isnull(df.iloc[hr_line, i]) or (df.iloc[hr_line, i]==0)):
            xls_info['hr_start'] = i
            break
    print('hr_start', xls_info['hr_start'])

    xls_info['hr_end'] = xls_info['time_num']
    for i in range(xls_info['time_num'], -1, -1):
        if not (pd.isnull(df.iloc[hr_line, i]) or (df.iloc[hr_line, i]==0)):
            xls_info['hr_end'] = i
            break
    print('hr_end', xls_info['hr_end'])

    xls_info['bis_start'] = 1
    for i in range(1, xls_info['time_num']+1):
        if not (pd.isnull(df.iloc[bis_line, i]) or (df.iloc[bis_line, i]==0)):
            xls_info['bis_start'] = i
            break
    print('bis_start', xls_info['bis_start'])

    xls_info['bis_end'] = xls_info['time_num']
    for i in range(xls_info['time_num'], -1, -1):
        if not (pd.isnull(df.iloc[bis_line, i]) or (df.iloc[bis_line, i]==0)):
            xls_info['bis_end'] = i
            break
    print('bis_end', xls_info['bis_end'])

    # 看起来心率在每个时刻都有记录，这里只查找血压的起始时间，后续画图使用心率的时间轴作为baseline
    xls_info['bp_start'] = 1
    xls_info['bp_end'] = xls_info['time_num']
    if bp_line != -1:
        for i in range(1, xls_info['time_num']+1):
            if not (pd.isnull(df.iloc[bp_line, i]) or (df.iloc[bp_line, i]==0)):
                xls_info['bp_start'] = i
                break
        for i in range(xls_info['time_num'], -1, -1):
            if not (pd.isnull(df.iloc[bp_line, i]) or (df.iloc[bp_line, i]==0)):
                xls_info['bp_end'] = i
                break
    print('bp_start', xls_info['bp_start'])  
    print('bp_end', xls_info['bp_end']) 

    xls_info['eeg_start_idx'] = 0
    for key in df.keys():
        if key == eeg_start_time[exp_id][-8:-3]:
            break
        xls_info['eeg_start_idx'] += 1
    if not eeg_start_time[exp_id][-2:] == '00':
        xls_info['eeg_start_idx'] += 1
    print('eeg_start_idx', xls_info['eeg_start_idx'])

    xls_info['eeg_end_idx'] = 0
    for key in df.keys():
        if key == eeg_end_time[exp_id][-8:-3]:
            break
        xls_info['eeg_end_idx'] += 1
    print('eeg_end_idx', xls_info['eeg_end_idx'])

    if hr_line == -1:
        raise Exception('No heart rate!!')
    # xls_info['hr'] = np.array(list(map(int, df.iloc[hr_line, 1:].tolist())))
    xls_info['hr'] = np.array(list(map(int, df.iloc[hr_line, xls_info['hr_start']:xls_info['hr_end']+1].tolist())))
    if bp_line == -1:
        # xls_info['bp'] = np.zeros(xls_info['hr'].shape[0]) - 1
        xls_info['bp'] = xls_info['hr'].copy()
    else:
        xls_info['bp'] = np.array(list(map(int, df.iloc[bp_line, xls_info['bp_start']:xls_info['bp_end']+1].tolist())))
    xls_info['bis'] = np.array(list(map(int, df.iloc[bis_line, xls_info['bis_start']:xls_info['bis_end']+1].tolist())))

    return xls_info

def find_event(df, df_event, csv_path, event):
    event_time = -1
    tmp = df_event[event]
    for i in range(tmp.shape[0]):
        if not pd.isnull(tmp[i]):
            tmp_tmp = tmp[i]
    # print('[debug]', tmp_tmp, event)
    # 0523那天的excel表格式有所不同，需要单独处理
    if '0523' in csv_path:
        tmp_hm = tmp_tmp[0:5] + ':00'
    else:
        tmp_hm = tmp_tmp[0:5]

    for key in df.keys():
        if str(key) == str(tmp_hm):
            break
        event_time += 1        
       
    # print('event_time', event_time)
    return event_time

def find_events(df, df_event, csv_path, event):
    event_time = []
    tmp = df_event[event]
    tmp_tmp = []
    for i in range(tmp.shape[0]):
        if not pd.isnull(tmp[i]):
            tmp_tmp.append(tmp[i])

    for i in range(len(tmp_tmp)):
        event_time_tmp = -1
        # 0523那天的excel表格式有所不同，需要单独处理
        if '0523' in csv_path:
            tmp_hm = tmp_tmp[i][0:5] + ':00'
        else:
            tmp_hm = tmp_tmp[i][0:5]

        for key in df.keys():
            if str(key) == str(tmp_hm):
                break
            event_time_tmp += 1   
        if not event_time_tmp in event_time:
            event_time.append(event_time_tmp)     
    # print('event_time', event_time)
    return event_time

def process_xls_new(csv_path, exp_id, eeg_start_time, eeg_end_time):
    xls_info = {}
    csv_list = os.listdir(csv_path)
    if '.DS_Store' in csv_list:
        csv_list.remove('.DS_Store')
    csv_list.sort()
    print('csv_list', csv_list)

    df = pd.read_excel('{}/{}'.format(csv_path, csv_list[exp_id]))
    # print(df)

    hr_line = -1
    bp_line = -1
    bis_line = -1
    for i in range(df.iloc[:,0].shape[0]):
        if df.iloc[i,0] == '心率':
            hr_line = i
        elif df.iloc[i,0] == '无创收缩压':
            bp_line = i
        elif df.iloc[i,0] == 'BIS':
            bis_line = i

    # 实际上在xls开始时的时间间隔可能不是1min，但这一段我们往往不关心，认为其间隔是1min也关系不大
    xls_info['time_num'] = len(df.keys()) - 1
    print('time_num', xls_info['time_num'])

    csv_list_all = os.listdir('{}/event'.format(csv_path))
    csv_list = []
    for item in csv_list_all:
        if '.csv' in item:
            csv_list.append(item)
    csv_list.sort()
    # print('[debug] csv_list', csv_list)
    df_event = pd.read_csv('{}/event/{}'.format(csv_path, csv_list[exp_id]))

    xls_info['time_loc'] = find_event(df, df_event, csv_path, '睫毛反射消失时间')
    xls_info['time_roc'] = find_event(df, df_event, csv_path, '睫毛反射出现时间')
    xls_info['time_inject'] = find_event(df, df_event, csv_path, '给药开始时间')
    if '0523' in csv_path:
        xls_info['time_druge'] = find_event(df, df_event, csv_path, '给药结束时间')
    else:
        xls_info['time_druge'] = find_event(df, df_event, csv_path, '停止给药时间')
    # xls_info['time_wake'] = find_event(df, df_event, csv_path, '患者清醒')
    xls_info['time_move'] = find_events(df, df_event, csv_path, '体动')
    xls_info['time_add_drug'] = find_events(df, df_event, csv_path, '4ug瑞芬')
    
    print('time_loc', xls_info['time_loc'])
    print('time_roc', xls_info['time_roc'])
    print('time_inject', xls_info['time_inject'])
    # print('time_wake', xls_info['time_wake'])

    xls_info['hr_start'] = 1
    for i in range(1, xls_info['time_num']+1):
        if not (pd.isnull(df.iloc[hr_line, i]) or (df.iloc[hr_line, i]==0)):
            xls_info['hr_start'] = i
            break
    print('hr_start', xls_info['hr_start'])

    xls_info['hr_end'] = xls_info['time_num']
    for i in range(xls_info['time_num'], -1, -1):
        if not (pd.isnull(df.iloc[hr_line, i]) or (df.iloc[hr_line, i]==0)):
            xls_info['hr_end'] = i
            break
    print('hr_end', xls_info['hr_end'])

    xls_info['bis_start'] = 1
    for i in range(1, xls_info['time_num']+1):
        if not (pd.isnull(df.iloc[bis_line, i]) or (df.iloc[bis_line, i]==0)):
            xls_info['bis_start'] = i
            break
    print('bis_start', xls_info['bis_start'])

    xls_info['bis_end'] = xls_info['time_num']
    for i in range(xls_info['time_num'], -1, -1):
        if not (pd.isnull(df.iloc[bis_line, i]) or (df.iloc[bis_line, i]==0)):
            xls_info['bis_end'] = i
            break
    print('bis_end', xls_info['bis_end'])

    # 看起来心率在每个时刻都有记录，这里只查找血压的起始时间，后续画图使用心率的时间轴作为baseline
    xls_info['bp_start'] = 1
    xls_info['bp_end'] = xls_info['time_num']
    if bp_line != -1:
        for i in range(1, xls_info['time_num']+1):
            if not (pd.isnull(df.iloc[bp_line, i]) or (df.iloc[bp_line, i]==0)):
                xls_info['bp_start'] = i
                break
        for i in range(xls_info['time_num'], -1, -1):
            if not (pd.isnull(df.iloc[bp_line, i]) or (df.iloc[bp_line, i]==0)):
                xls_info['bp_end'] = i
                break
    print('bp_start', xls_info['bp_start'])  
    print('bp_end', xls_info['bp_end']) 

    xls_info['eeg_start_idx'] = 0
    for key in df.keys():
        # print('key', key)
        if '0523' in csv_path:
            key = str(key)[0:-3]
            # print('[debug] key', key, eeg_start_time[exp_id][-8:-3])
        if key == eeg_start_time[exp_id][-8:-3]:
            break
        xls_info['eeg_start_idx'] += 1
    if not eeg_start_time[exp_id][-2:] == '00':
        xls_info['eeg_start_idx'] += 1
    print('eeg_start_idx', xls_info['eeg_start_idx'])

    xls_info['eeg_end_idx'] = 0
    for key in df.keys():
        if '0523' in csv_path:
            key = str(key)[0:-3]
        if key == eeg_end_time[exp_id][-8:-3]:
            break
        xls_info['eeg_end_idx'] += 1
    print('eeg_end_idx', xls_info['eeg_end_idx'])

    if hr_line == -1:
        raise Exception('No heart rate!!')
    # xls_info['hr'] = np.array(list(map(int, df.iloc[hr_line, 1:].tolist())))
    xls_info['hr'] = np.array(list(map(int, df.iloc[hr_line, xls_info['hr_start']:xls_info['hr_end']+1].tolist())))
    if bp_line == -1:
        # xls_info['bp'] = np.zeros(xls_info['hr'].shape[0]) - 1
        xls_info['bp'] = xls_info['hr'].copy()
    else:
        xls_info['bp'] = np.array(list(map(int, df.iloc[bp_line, xls_info['bp_start']:xls_info['bp_end']+1].tolist())))
    xls_info['bis'] = np.array(list(map(int, df.iloc[bis_line, xls_info['bis_start']:xls_info['bis_end']+1].tolist())))

    return xls_info

def remove_tail(x_ori, y_ori):
    cnt_tmp = 0
    for i in range(y_ori.shape[0]-1, -1, -1):
        if y_ori[i] == 0:
            cnt_tmp += 1
        else:
            break
    if cnt_tmp==0:
        y = y_ori
        x = x_ori
    else:
        x = x_ori[0:-cnt_tmp]
        y = y_ori[0:-cnt_tmp]
    return x, y

def draw_results(exp_date, exp, xls_info, eeg_feature_dict):
    color_bkp = ['forestgreen', 'darkorange',  'dimgray',
                    'limegreen', 'royalblue', 'darkgrey', 'forestgreen', 'darkblue', 'purple']
    colors = ['darkgrey', 'k', 'firebrick', 'forestgreen', 'darkorange', 'royalblue', 'purple','limegreen', 'dimgray', 'darkblue']

    plt.figure(figsize=(12,4))
    # hr_x = np.arange(xls_info['time_num']) + 1
    hr_x = np.arange(xls_info['hr_end']-xls_info['hr_start']+1)+(xls_info['hr_start'])
    hr_x, hr_y = remove_tail(hr_x, xls_info['hr'])
    plt.plot(hr_x, hr_y, label='HR', color=colors[0], lw=2)

    
    # bp_x = np.arange(xls_info['time_num']-(xls_info['bp_start']-1))+(xls_info['bp_start']-1)
    bp_x = np.arange(xls_info['bp_end']-xls_info['bp_start']+1)+(xls_info['bp_start'])
    bp_x, bp_y = remove_tail(bp_x, xls_info['bp'])
    plt.plot(bp_x, bp_y, label='BP', color=colors[1], lw=2)

    # bis_x = np.arange(xls_info['time_num']-(xls_info['bis_start']))+(xls_info['bis_start'])
    bis_x = np.arange(xls_info['bis_end']-xls_info['bis_start']+1)+(xls_info['bis_start'])
    bis_x, bis_y = remove_tail(bis_x, xls_info['bis'])
    plt.plot(bis_x, bis_y, label='BIS', color=colors[2], lw=2)

    eeg_feature_x = np.arange(xls_info['eeg_end_idx']-xls_info['eeg_start_idx'])+(xls_info['eeg_start_idx'])
    # print('[debug] eeg_feature_x', eeg_feature_x)
    # 有可能因为丢包，EEG会比预想的少1min
    if eeg_feature_dict['PE'].shape[0] < eeg_feature_x.shape[0]:
        eeg_feature_x = eeg_feature_x[0:-1]
    # print('[debug] eeg_feature_x', eeg_feature_x)
    cnt_tmp = 2
    for key in eeg_feature_dict.keys():
        cnt_tmp += 1
        # 有可能EEG实际上结束的比excel最后一个时间要晚
        plt.plot(eeg_feature_x, eeg_feature_dict[key][0:eeg_feature_x.shape[0]], label=key, color=colors[cnt_tmp], lw=1)

    if xls_info['time_loc'] >= 0:
        plt.axvline(xls_info['time_loc'], label='LOC', color='r', ls='dashed', lw=2)
    if xls_info['time_roc'] >= 0:
        plt.axvline(xls_info['time_roc'], label='ROC', color='b', ls='dashed', lw=2)

    # plt.ylim(np.min(xls_info['hr'])-5, np.max(xls_info['hr'])+5)
    plt.xticks(font={'family':'Arial', 'size':16})
    plt.yticks(font={'family':'Arial', 'size':16})
    plt.xlabel('Time (minute)', font={'family':'Arial', 'size':20})
    plt.ylabel('Values', font={'family':'Arial', 'size':20})
    plt.title('Exp_{}_{}'.format(exp_date, exp), font={'family':'Arial', 'size':24})

    plt.legend(loc=(1.05, 0), edgecolor='k', prop={'family':'Arial', 'size':14})
    plt.savefig('figs/{}/features_exp_{}.png'.format(exp_date, exp), dpi=300, bbox_inches='tight')
    plt.show()

    corr_eeg_features = np.zeros((2, len(eeg_feature_dict.keys())+1))
    p_eeg_features = np.zeros((2, len(eeg_feature_dict.keys())+1))

    com = [x for x in hr_x if x in bis_x]
    hr_idx = com - hr_x[0]
    bis_idx = com - bis_x[0]
    corr_eeg_features[0, 0], p_eeg_features[0, 0] = pearsonr(bis_y[bis_idx], xls_info['hr'][hr_idx])
    # tmp = np.max(bis_x-np.min(hr_x)) - xls_info['hr'].shape[0]
    # print('tmp', tmp)
    # if tmp < 0:
    #     hr_idx = bis_x-np.min(hr_x)
    #     corr_eeg_features[0, 0], p_eeg_features[0, 0] = pearsonr(bis_y, xls_info['hr'][hr_idx])
    # else:
    #     hr_idx = bis_x-np.min(hr_x)
    #     hr_idx = hr_idx[0:-(tmp+1)]
    #     corr_eeg_features[0, 0], p_eeg_features[0, 0] = pearsonr(bis_y[0:-(tmp+1)], xls_info['hr'][hr_idx])
    com = [x for x in bp_x if x in bis_x]
    bp_idx = com - bp_x[0]
    bis_idx = com - bis_x[0]
    # print('[debug] bis_idx', bis_idx, bis_idx.shape)
    # print('[debug] bis_y', bis_y, bis_y.shape)
    # print('[debug] bp_idx', bp_idx, bp_idx.shape)
    corr_eeg_features[1, 0], p_eeg_features[1, 0] = pearsonr(bis_y[bis_idx], xls_info['bp'][bp_idx])

    # tmp = np.max(bis_x-np.min(bp_x)) - xls_info['bp'].shape[0]
    # if tmp < 0:
    #     bp_idx = bis_x-np.min(bp_x)
    #     corr_eeg_features[1, 0], p_eeg_features[1, 0] = pearsonr(bis_y, xls_info['bp'][bp_idx])
    # else:
    #     bp_idx = bis_x-np.min(bp_x)
    #     bp_idx = bp_idx[0:-(tmp+1)]
    #     corr_eeg_features[1, 0], p_eeg_features[1, 0] = pearsonr(bis_y[0:-(tmp+1)], xls_info['bp'][bp_idx])
    cnt_tmp = 0
    for key in eeg_feature_dict.keys():
        cnt_tmp += 1
        com = [x for x in hr_x if x in eeg_feature_x]
        hr_idx = com - hr_x[0]
        eeg_idx = com - eeg_feature_x[0]
        corr_eeg_features[0, cnt_tmp], p_eeg_features[0, cnt_tmp] = pearsonr(eeg_feature_dict[key][eeg_idx], 
                                                                             xls_info['hr'][hr_idx])
        # corr_eeg_features[0, cnt_tmp], p_eeg_features[0, cnt_tmp] = pearsonr(eeg_feature_dict[key][0:eeg_feature_x.shape[0]], 
        #                                                                      xls_info['hr'][eeg_feature_x-np.min(hr_x)])
        com = [x for x in bp_x if x in eeg_feature_x]
        bp_idx = com - bp_x[0]
        eeg_idx = com - eeg_feature_x[0]
        corr_eeg_features[1, cnt_tmp], p_eeg_features[1, cnt_tmp] = pearsonr(eeg_feature_dict[key][eeg_idx], 
                                                                             xls_info['bp'][bp_idx])
        # corr_eeg_features[1, cnt_tmp], p_eeg_features[1, cnt_tmp] = pearsonr(eeg_feature_dict[key][0:eeg_feature_x.shape[0]], 
        #                                                                      xls_info['bp'][eeg_feature_x-np.min(bp_x)])

    color = ['firebrick', 'royalblue']
    ecolor = ['k', 'k']
    width = 0.3
    
    x_label = ['BIS']
    for key in eeg_feature_dict.keys():
        x_label.append(key)
        
    x_bar = np.arange(len(x_label))

    plt.figure(figsize=(6, 4))
    plt.bar(x_bar-width-0.05, corr_eeg_features[0,:], label='Corr vs HR',
            width=width, edgecolor=ecolor[0], color=color[0], alpha=1, zorder=100)
    plt.bar(x_bar+0.05, corr_eeg_features[1,:], label='Corr vs BP',
            width=width, edgecolor=ecolor[1], color=color[1], alpha=1, zorder=100)

    for i in range(p_eeg_features.shape[0]):
        for j in range(p_eeg_features.shape[1]):
            if p_eeg_features[i,j] < 0.01:
                print(p_eeg_features[i,j])
                if i==0:
                    x_tmp = j - width - 0.24
                else:
                    x_tmp = j - 0.1
                if corr_eeg_features[i,j] > 0:
                    y_tmp = corr_eeg_features[i,j]
                else:
                    y_tmp = 0
                plt.annotate(r'$**$', xy=(x_tmp, y_tmp), fontsize=16, color='r')
            elif p_eeg_features[i,j] < 0.05:
                print(p_eeg_features[i,j])
                if i==0:
                    x_tmp = j - width - 0.14
                else:
                    x_tmp = j - 0.04
                if corr_eeg_features[i,j] > 0:
                    y_tmp = corr_eeg_features[i,j]
                else:
                    y_tmp = 0
                plt.annotate(r'$*$', xy=(x_tmp, y_tmp), fontsize=16, color='r')

    plt.bar(x_bar-width/2, np.zeros(x_bar.shape[0]), tick_label=x_label)
    plt.xticks(fontproperties = 'Arial', size = 14)
    plt.yticks(fontproperties = 'Arial', size = 14)
    plt.ylabel('Corr', font={'family':'Arial', 'size':16})
    plt.xlabel('Features', font={'family':'Arial', 'size':16})
    
    plt.ylim(np.min(corr_eeg_features) - 0.05, np.max(corr_eeg_features) + 0.05)
    plt.legend(ncol=1, edgecolor='k', prop={'family':'Arial', 'size':14})
    # plt.grid(zorder=0)
    plt.tight_layout()
    plt.savefig('figs/{}/corr_exp_{}.png'.format(exp_date, exp), dpi=300, bbox_inches='tight')
    plt.show()

def overlap_ratio(x_low, x_high):
    cnt = 0
    idx_low = 0
    for i in range(x_high.shape[0]):
        if x_high[i] > 0:
            idx_low = i
            break
    idx_high = x_low.shape[0]-1
    idx_high = np.max(x_low)
    for i in range(x_low.shape[0]-1, -1, -1):
        if x_low[i] > 0:
            idx_high = i
            break
    # print(idx_high, idx_low)
    for i in range(int(idx_low), int(idx_high) + 1):
        cnt += x_low[i]
        cnt += x_high[i]
    return cnt / (np.sum(x_low) + np.sum(x_high))

def draw_discrimination(eeg_feature_dict, xls_info, key, exp_date, exp):
    label = []
    feature = []
    bis_x = np.arange(xls_info['bis_end']-xls_info['bis_start']+1)+(xls_info['bis_start'])
    eeg_feature_x = np.arange(xls_info['eeg_end_idx']-xls_info['eeg_start_idx'])+(xls_info['eeg_start_idx'])
    # 有可能因为丢包，EEG会比预想的少1min
    if eeg_feature_dict['PE'].shape[0] < eeg_feature_x.shape[0]:
        eeg_feature_x = eeg_feature_x[0:-1]
    # print(eeg_feature_x)
    if key=='bis':
        draw_x = bis_x
        draw_y = xls_info['bis']
    else:
        draw_x = eeg_feature_x
        draw_y = eeg_feature_dict[key]
    grid_unit = 5
    bin_num = math.ceil(np.max(draw_y) / grid_unit + 0.001)
    print('bin_num', bin_num)
    draw_c = np.zeros(bin_num)
    draw_nc = np.zeros(bin_num)
    for i in range(draw_x.shape[0]):
        if draw_x[i] < xls_info['time_loc']:
            draw_c[int(draw_y[i]//grid_unit)] += 1
            label.append(1)
            feature.append(draw_y[i])
        elif xls_info['time_loc'] <= draw_x[i] < xls_info['time_roc']:
            draw_nc[int(draw_y[i]//grid_unit)] += 1
            label.append(0)
            feature.append(draw_y[i])
        elif draw_x[i] >= xls_info['time_roc']:
            draw_c[int(draw_y[i]//grid_unit)] += 1
            label.append(1)
            feature.append(draw_y[i])
        else:
            raise Exception('Error')
    x_bar = np.arange(bin_num) * grid_unit
    width = 0.4 * grid_unit
    plt.bar(x_bar-width/2, draw_c, width=width, color='firebrick', label='C')
    plt.bar(x_bar+width/2, draw_nc, width=width, color='royalblue', label='NC')

    plt.xticks(fontproperties = 'Arial', size = 14)
    plt.yticks(fontproperties = 'Arial', size = 14)
    plt.ylabel('Number', font={'family':'Arial', 'size':16})
    plt.xlabel('Feature Value', font={'family':'Arial', 'size':16})
    plt.title('{}'.format(key), font={'family':'Arial', 'size':24})

    plt.legend(ncol=1, edgecolor='k', prop={'family':'Arial', 'size':14})
    plt.tight_layout()
    plt.savefig('figs/{}/{}_{}.png'.format(exp_date, exp, key), dpi=300, bbox_inches='tight')
    plt.show()
    return draw_c, draw_nc, feature, label