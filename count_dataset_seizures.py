import glob
from utilities import manifold_utilities


if __name__ == "__main__":

    atd_file = '/media/graham/MOBO_RAID0/Ubuntu_Projects/SEEG_Tornados/data/all_time_data_01092023_112957.csv'

    # dataset_dir = '/media/graham/MOBO_RAID0/Ubuntu_Projects/SEEG_Tornados/bse_inference/train45/latent_files/64SecondWindow_64SecondStride'
    dataset_dir = '/media/graham/MOBO_RAID0/Ubuntu_Projects/SEEG_Tornados/bse_inference/val13/latent_files/64SecondWindow_64SecondStride'

    all_files = glob.glob(f"{dataset_dir}/*.pkl")
    unique_pats = list(set([x.split("/")[-1].split("_")[0] for x in all_files]))

    seiz_count = 0

    for pat in unique_pats:
        _, _, seiz_type = manifold_utilities.get_pat_seiz_datetimes(pat, atd_file)
        seiz_count = seiz_count + len(seiz_type)

    print(f"Dataset total num seizures: {seiz_count}")