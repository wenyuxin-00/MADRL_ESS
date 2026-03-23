"""濂栧姳鍑芥暟鍩虹被銆?

瀹氫箟鎵€鏈夊鍔卞嚱鏁扮殑缁熶竴鎺ュ彛銆?

涓昏绫?
    BaseReward -- 濂栧姳鍑芥暟鎶借薄鍩虹被
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
import numpy as np


@dataclass
class ComponentMeta:
    """濂栧姳鍒嗛噺鐨勫厓鏁版嵁锛岄┍鍔?history 鍔ㄦ€佸垵濮嬪寲鍜岀敾鍥捐嚜閫傚簲銆?

    灞炴€?
        key:   瀵瑰簲 info dict 鐨勯敭鍚嶏紝濡?"r_inc"
        label: 瀛愬浘鏍囬锛屽 "+ r_inc (incremental cost)"
        color: 瀛愬浘棰滆壊锛屽 "green"
        sign:  +1 鎴?-1锛堣鍒嗛噺鍦?total 涓殑璐＄尞绗﹀彿锛岀敤浜庢纭疮鍔犲巻鍙诧級
               +1 鈫?total 涓槸鍔犲彿锛?1 鈫?total 涓槸鍑忓彿
    """
    key: str       # info 瀛楀吀涓殑閿悕
    label: str     # 鍙鍖栧瓙鍥炬爣棰?
    color: str     # 鍙鍖栧瓙鍥鹃鑹?
    sign: int      # 鍦ㄦ€诲鍔变腑鐨勭鍙凤細+1 琛ㄧず姝ｈ础鐚紝-1 琛ㄧず璐熻础鐚紙鎯╃綒椤癸級


class RewardFn(ABC):
    """鎵€鏈夊鍔卞嚱鏁扮殑鎶借薄鍩虹被銆?

    瀛愮被闇€瀹炵幇涓や釜鎺ュ彛锛?
    1. compute()         鈥?璁＄畻濂栧姳锛岃繑鍥?(total, components) 鍏冪粍
    2. component_meta   鈥?澹版槑鍚勫垎閲忕殑鍏冩暟鎹紙椹卞姩鍔ㄦ€佺敾鍥惧拰 history锛?

    env_state 閿鏄庯細
        e_bat_req  (N,)  鈥?璇锋眰鍔熺巼锛堝彲鑳借秺闄愶級
        e_bat      (N,)  鈥?鎵ц鍔熺巼锛堝彲琛屽煙鎶曞奖鍚庯級
        soc_t      (N,)  鈥?鏈寮€濮嬫椂鐨?SoC
        soc_next   (N,)  鈥?鏈缁撴潫鍚庣殑 SoC
        e_t        (N,)  鈥?鏈寮€濮嬫椂鐨勫偍鑳介噺 = soc_t * c_bat
        e_next     (N,)  鈥?鏈缁撴潫鍚庣殑鍌ㄨ兘閲?= soc_next * c_bat
        price_t    float 鈥?褰撳墠鐢典环
        load_t     (N,)  鈥?褰撳墠璐熻嵎
        mu_t       float 鈥?鏈潵 K 姝ュ钩鍧囩數浠凤紙褰撳墠姝ワ級
        mu_next    float 鈥?鏈潵 K 姝ュ钩鍧囩數浠凤紙涓嬩竴姝ワ級
        gamma      float 鈥?鎶樻墸鍥犲瓙锛圥BRS 浣跨敤锛?
    """

    @abstractmethod
    def compute(self, env_state: dict) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        """
        璁＄畻濂栧姳骞惰繑鍥炲悇鍒嗛噺銆?

        Returns:
            (total_reward, components)
            - total_reward: np.ndarray, shape (N,), dtype float32
            - components:   dict[str, np.ndarray], 姣忎釜鍊?shape (N,) float32
                            閿悕涓?component_meta 涓殑 key 涓€涓€瀵瑰簲
                            瀛樺偍鐨勬槸"缁濆鍊?鈥斺€旂鍙风敱 ComponentMeta.sign 绠＄悊
        """
        ...

    @property
    @abstractmethod
    def component_meta(self) -> list[ComponentMeta]:
        """
        杩斿洖鍚勫垎閲忕殑鍏冩暟鎹垪琛紙list[ComponentMeta]锛夈€?
        椤哄簭涓?compute() 杩斿洖鐨?components dict 涓殑閿『搴忎竴鑷淬€?
        璇ュ垪琛ㄩ┍鍔細
          - grid_env.py step() 涓?info 鐨勫姩鎬佹瀯寤?
          - Runner 涓?history dict 鐨勫姩鎬佸垵濮嬪寲
          - Runner 涓瘡姝ユ暟鎹殑鍔ㄦ€佸瓨鍌?
          - plot_reward_decomposition 鐨勫瓙鍥炬暟閲忋€侀鑹层€佹爣棰?
        """
        ...

