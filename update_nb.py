import json

with open('examples/2pcf/scope_paper_plots.ipynb', 'r') as f:
    nb = json.load(f)

new_cells = []
new_cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "## Model Robustness: Millennium I and II\n\nValidating that SCOPE works across different box sizes and resolutions."
    ]
})
new_cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "def plot_model_robustness():\n    cfgs = [\n        {'sim': 'Mill1', 'iz': 155, 'mtag': 'mstar9.0', 'data': Path('../../data/2pcf/scope_xi_mill1')},\n        {'sim': 'Mill2', 'iz': 40, 'mtag': 'mstar_none', 'data': Path('../../data/2pcf/scope_xi_mill2')}\n    ]\n    \n    fig, ax = plt.subplots(figsize=(8, 6))\n    for cfg in cfgs:\n        if not cfg['data'].exists(): continue\n        g = load_and_aggregate(cfg['iz'], cfg['mtag'], mode='xi', sim=cfg['sim'], data_root=cfg['data'])\n        if g is None or 'frac_diff_corr' not in g.columns: continue\n        \n        sub = g[g['n_subvol'] == 64].sort_values('r_val')\n        if not sub.empty:\n            ax.plot(sub['r_val'], sub['frac_diff_corr'], label=cfg['sim'] + ' (N=64)')\n            \n    ax.set_xscale('log'); ax.axhline(0, color='k', alpha=0.3)\n    ax.set_ylabel(r'$\\Delta\\xi / \\xi$'); ax.set_ylim(-0.15, 0.15)\n    ax.set_title('Robustness across simulation suites')\n    ax.legend(); plt.show()\n\nplot_model_robustness()"
})

nb['cells'].extend(new_cells)

with open('examples/2pcf/scope_paper_plots.ipynb', 'w') as f:
    json.dump(nb, f, indent=1)
