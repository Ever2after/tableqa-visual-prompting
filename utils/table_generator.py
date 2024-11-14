import matplotlib
matplotlib.use('Agg')  # 비-GUI 백엔드 사용
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib import rcParams
from matplotlib import font_manager as fm
from matplotlib.colors import LinearSegmentedColormap

import base64
import io

custom_cmap = LinearSegmentedColormap.from_list(
    'custom_white_yellow', ['#FFFFFF', '#FFFF00'], N=256
)

# font_path = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
# fontprop = fm.FontProperties(fname=font_path, size=10)

# rcParams['font.sans-serif'] = fontprop.get_name()
rcParams['axes.unicode_minus'] = False  # '-' 기호 깨짐 방지

# 수식 해석 비활성화 설정
rcParams['text.usetex'] = False  # LaTeX 수식 해석 비활성화
rcParams['text.parse_math'] = False  # 수식 파싱 해석 비활성화

def generate_table_image(table_df, score_df, cmap='YlOrBr'):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axis('tight')
    ax.axis('off')

    # Create table
    table = ax.table(cellText=table_df.values, colLabels=table_df.columns, cellLoc='center', loc='center')

    # Style and color settings
    table.auto_set_font_size(False)
    table.set_fontsize(10)

    # Adjust column widths based on content
    for col_idx in range(len(table_df.columns)):
        table.auto_set_column_width([col_idx])  # Set width based on column content

    if score_df is not None:
        norm = Normalize(vmin=score_df.min().min(), vmax=score_df.max().max())
        if type(cmap) == str:
            try:
                colormap = plt.colormaps[cmap]  # 최신 colormap 호출 방식
            except:
                colormap = cmap
        else:
            colormap = cmap
        for (i, j), cell in table.get_celld().items():
            if i == 0:  # Header row
                cell.set_facecolor('#CCCCCC')
                cell.set_text_props(weight='bold')
            else:
                color = colormap(norm(score_df.iloc[i - 1, j]))
                cell.set_facecolor(color)
                cell.set_text_props(color='black')
    else:
        for (i, j), cell in table.get_celld().items():
            if i == 0:
                cell.set_facecolor('#CCCCCC')
                cell.set_text_props(weight='bold')

    # Save to buffer
    img_buffer = io.BytesIO()
    plt.savefig(img_buffer, format='jpeg', bbox_inches='tight', dpi=100)
    plt.close(fig)

    # Convert to Base64
    img_buffer.seek(0)
    img_base64 = base64.b64encode(img_buffer.read()).decode('utf-8')
    img_buffer.close()
    return img_base64

