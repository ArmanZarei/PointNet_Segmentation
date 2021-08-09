# PointNet Segmentation

After training the model for `15` epochs:
<table style="text-align: center;">
    <thead>
        <tr>
            <th>Train Set</th>
            <th>Validation Set</th>
            <th>Test Set</th>
        </tr>
    </thead>
    <tr>
        <td>91%</td>
        <td>90%</td>
        <td>90%</td>
    </tr>
</table>

![Training Process](images/training.png)

<table>
    <thead>
        <tr>
            <th style="text-align: center;">Ground Truth</th>
            <th style="text-align: center;">Predicted</th>
            <th style="text-align: center;">Diff.</th>
        </tr>
    </thead>
    <tr>
        <td><img src='images/labeled_0.gif'></td>
        <td><img src='images/predicted_0.gif'></td>
        <td><img src='images/diff_0.gif'></td>
    </tr>
    <tr>
        <td><img src='images/labeled_1.gif'></td>
        <td><img src='images/predicted_1.gif'></td>
        <td><img src='images/diff_1.gif'></td>
    </tr>
    <tr>
        <td><img src='images/labeled_2.gif'></td>
        <td><img src='images/predicted_2.gif'></td>
        <td><img src='images/diff_2.gif'></td>
    </tr>
</table>