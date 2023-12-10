from first_code import bhogen

if __name__ == "__main__":
    cluster = bhogen(Nbh=21000, n_mus=1000, spin_dis="unif")
    cluster.populate()
    cluster.build_mass_grid()
    cluster.emit_GW()
