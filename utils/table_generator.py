import matplotlib
matplotlib.use('Agg')  # 비-GUI 백엔드 사용

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, LinearSegmentedColormap
from matplotlib import rcParams
import base64
import io

# 사용자 정의 컬러맵
custom_cmap = LinearSegmentedColormap.from_list(
    'custom_white_yellow', ['#FFFFFF', '#FFFF00'], N=256
)

# rcParams 설정
rcParams['axes.unicode_minus'] = False  # '-' 기호 깨짐 방지
rcParams['text.usetex'] = False  # LaTeX 수식 해석 비활성화

def generate_table_image(table_df, score_df=None, cmap=custom_cmap, dpi=100):
    """
    테이블과 선택적 스코어 데이터를 기반으로 테이블 이미지를 생성하고 Base64로 인코딩.

    Parameters:
        - table_df (pd.DataFrame): 렌더링할 테이블 데이터프레임
        - score_df (pd.DataFrame or None): 셀 강조를 위한 스코어 데이터프레임
        - cmap (str or Colormap): 컬러맵 (기본: custom_cmap)
        - dpi (int): 이미지 해상도

    Returns:
        - img_base64 (str): Base64로 인코딩된 이미지 문자열
    """
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axis('tight')
    ax.axis('off')

    # 테이블 생성
    table = ax.table(cellText=table_df.values, colLabels=table_df.columns, cellLoc='center', loc='center')
    
    # 기본 스타일 설정
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.auto_set_column_width(list(range(len(table_df.columns))))

    # 스코어에 따른 컬러맵 적용
    if score_df is not None:
        norm = Normalize(vmin=score_df.min().min(), vmax=score_df.max().max())
        colormap = plt.cm.get_cmap(cmap) if isinstance(cmap, str) else cmap

        for (i, j), cell in table.get_celld().items():
            if i == 0:  # Header row
                cell.set_facecolor('#CCCCCC')
                cell.set_text_props(weight='bold')
            else:
                color = colormap(norm(score_df.iloc[i - 1, j]))
                cell.set_facecolor(color)
                cell.set_text_props(color='black')
    else:
        # 스코어가 없을 경우 기본 스타일 적용
        for (i, j), cell in table.get_celld().items():
            if i == 0:  # Header row
                cell.set_facecolor('#CCCCCC')
                cell.set_text_props(weight='bold')

    # 이미지 버퍼에 저장
    img_buffer = io.BytesIO()
    try:
        plt.savefig(img_buffer, format='jpeg', bbox_inches='tight', dpi=dpi)
        img_buffer.seek(0)
        img_base64 = base64.b64encode(img_buffer.read()).decode('utf-8')
    finally:
        img_buffer.close()
        plt.close(fig)

    return img_base64
