# Android开发

## TextView

```xml
 <TextView
        android:id="@+id/tv_one"
        android:text="@string/tv_one"
        android:textColor="#ffff0000"
        android:textStyle="italic"
        android:textSize="30sp"
        android:gravity="center"
        android:background="#ff00ff00"
        android:layout_width="200dp"
        android:layout_height="200dp"
        android:shadowColor="@color/black"
        android:shadowRadius="3.0"
        android:shadowDx="10.0"
        android:shadowDy="10.0"
        >
    </TextView>
```

### java获取id修改属性

```java
TextView tv_one = findViewById(R.id.tv_one);
tv_one.setText("hhh");
```

### 跑马灯效果

```xml
<TextView
        android:id="@+id/tv_one"
        android:text="@string/tv_one"
        android:textColor="#ffff0000"
        android:textStyle="italic"
        android:textSize="30sp"
        android:gravity="center"
        android:background="#ff00ff00"
        android:layout_width="match_parent"
        android:layout_height="200dp"
        android:shadowColor="@color/black"
        android:shadowRadius="3.0"
        android:shadowDx="10.0"
        android:shadowDy="10.0"

        android:singleLine="true"
        android:focusable="true"
        android:focusableInTouchMode="true"
        android:marqueeRepeatLimit="marquee_forever"
        android:ellipsize="marquee">
        <requestFocus/>
    </TextView>
```

## Button

```xml
<Button
        android:layout_width="match_parent"
        android:layout_height="match_parent"
        android:backgroundTint="@color/btn_color_selector"
        android:background="@drawable/btn_selector"
        android:text="我是按钮">
    </Button>
```

### 触发点击事件

```java
Button button = findViewById(R.id.btn);
        //点击事件
        button.setOnClickListener(new View.OnClickListener() {
            public void onClick(View view) {
                Log.e(TAG,"onclick");
                return ;
            }
        });
        button.setOnLongClickListener(new View.OnLongClickListener() {
            public boolean onLongClick(View view) {
                Log.e(TAG,"onlongclick");
                return false;
            }
        });
        //触碰事件
        button.setOnTouchListener(new View.OnTouchListener() {
            public boolean onTouch(View view, MotionEvent motionEvent) {
                Log.e(TAG,"ontouchclick");
                return false;
            }
        });
```

## EditText

```xml
 <EditText
        android:id="@+id/password"
        android:hint="请输入密码"
        android:drawableLeft="@drawable/ic_baseline_person_24"
        android:drawablePadding="20dp"
        android:textColorHint="#95a1aa"
        android:inputType="textPassword"
        android:layout_width="200dp"
        android:layout_height="100dp">
    </EditText>
```

## ImageView

```xml
<ImageView
        android:src="@drawable/plane"
        android:scaleType="fitXY"
        android:layout_width="match_parent"
        android:layout_height="100dp">
    </ImageView>
```

## ProgressBar

```xml
<ProgressBar
        android:id="@+id/bar"
        android:layout_width="wrap_content"
        android:layout_height="wrap_content">
    </ProgressBar>
    
    <ProgressBar
        android:id="@+id/horbar"
        style="?android:attr/progressBarStyleHorizontal"
        android:max="100"
        android:layout_width="match_parent"
        android:layout_height="wrap_content">
    </ProgressBar>
```

## Notification

```java
protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        manager = (NotificationManager) getSystemService(NOTIFICATION_SERVICE);
        if(Build.VERSION.SDK_INT>=Build.VERSION_CODES.O){
           NotificationChannel channel= new NotificationChannel("libra","测试通知",NotificationManager.IMPORTANCE_HIGH);
            manager.createNotificationChannel(channel);
        }
        notification=new NotificationCompat.Builder(this,"libra")
                .setContentTitle("官方通知")
                .setContentText("大事件")
                .setSmallIcon(R.drawable.ic_baseline_accessibility_24)
                .setLargeIcon(BitmapFactory.decodeResource(getResources(),R.drawable.plane))
                .setAutoCancel(true)
                .build();
    }
    
    public void sendNotification(View view) {
        manager.notify(1,notification);  //前面的1表示id
    }

    public void cancelNotification(View view) {
        manager.cancel(1);
    }
```

## Toolbar

```xml
<androidx.appcompat.widget.Toolbar
        android:id="@+id/tb"
        android:layout_width="match_parent"
        app:navigationIcon="@drawable/ic_baseline_arrow_back_24"
        app:title="标题"
        app:titleTextColor="#ff0000"
        app:titleMarginStart="90dp"
        app:logo="@mipmap/ic_launcher"
        android:layout_height="?attr/actionBarSize">
    </androidx.appcompat.widget.Toolbar>
```

## AlertDialog

```java
//点击按钮实现的方法
public void alert(View view) {
        AlertDialog.Builder builder = new AlertDialog.Builder(this);
        builder.setIcon(R.mipmap.ic_launcher)
                .setTitle("我是对话框")
                .setMessage("今天天气怎么样啊")
                .setPositiveButton("确定",new DialogInterface.OnClickListener(){
                    public void onClick(DialogInterface dialogInterface, int i) {

                    }
                })
                .setNegativeButton("取消",new DialogInterface.OnClickListener(){
                    public void onClick(DialogInterface dialogInterface,int i) {

                    }
                })
                .create()
                .show();
    }
//弹出提示
Toast.makeText(this,"请输入。。。"，Toast.LENGTH_SHORT).show()
```

## PopupView

```java
public void popWind(View view) {
        View popupView=getLayoutInflater().inflate(R.layout.popup_view,null);  //popup_view布局自己设定

        PopupWindow popupWindow = new PopupWindow(popupView, ViewGroup.LayoutParams.WRAP_CONTENT,
                ViewGroup.LayoutParams.WRAP_CONTENT,true);
        popupWindow.showAsDropDown(view);
    }
```

布局

## TableLayout

```xml
<TableLayout
        android:layout_width="match_parent"
        android:layout_height="200dp">
        <TableRow>

            <Button
                android:layout_width="wrap_content"
                android:layout_height="wrap_content"
                android:text="表格"></Button>

            <Button
                android:layout_width="wrap_content"
                android:layout_height="wrap_content"
                android:text="表格"></Button>
        </TableRow>
    </TableLayout>
```

## GridLayout

```xml
<GridLayout
        android:layout_width="match_parent"
        android:layout_height="match_parent">

        <Button
            android:layout_width="wrap_content"
            android:layout_height="wrap_content"
            android:layout_row="1"
            android:layout_rowSpan="2"
            android:layout_column="1"
            android:text="网格1"></Button>

        <Button
            android:layout_width="wrap_content"
            android:layout_height="wrap_content"
            android:layout_row="0"
            android:layout_column="0"
            android:layout_columnSpan="2"
            android:text="网格2"></Button>
    </GridLayout>
```

ListView  列表展示

adapter

```java
package com.example.an_w;

import android.content.Context;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.BaseAdapter;
import android.widget.TextView;

import java.util.List;

public class MyAdaper extends BaseAdapter {
    private List<Bean> data;
    private Context context;

    public MyAdaper() {
    }

    public MyAdaper(List<Bean> data, Context context) {
        this.data = data;
        this.context = context;
    }

    @Override
    public int getCount() {
        return data.size();
    }

    @Override
    public Object getItem(int i) {
        return null;
    }

    @Override
    public long getItemId(int i) {
        return i;
    }

    @Override
    public View getView(int i, View view, ViewGroup viewGroup) {
        ViewHolder viewHolder;
        if(view==null)
        {
            viewHolder=new ViewHolder();
            view=LayoutInflater.from(context).inflate(R.layout.list_item,viewGroup,false);
            viewHolder.textView=view.findViewById(R.id.tv);
            view.setTag(viewHolder);
        }
        viewHolder.textView.setText(data.get(i).getName());
        return view;
    }
    //这样可以封装多个组件
    private final class ViewHolder{
        TextView textView;
    }
}
```

### mainactivity

```java
for(int i=0;i<5;i++)
        {
            Bean bean=new Bean();
            bean.setName("libra"+i);
            data.add(bean);
        }
        ListView listView=findViewById(R.id.lv);
        listView.setAdapter(new MyAdaper(data,this));
        listView.setOnItemClickListener(new AdapterView.OnItemClickListener() {
            @Override
            public void onItemClick(AdapterView<?> adapterView, View view, int i, long l) {
                Log.e("LOE","sdfsd"+i);
            }
        });
```

```xml
<ListView
        android:id="@+id/lv"
        android:layout_width="match_parent"
        android:layout_height="match_parent">
</ListView>
```

## Activity启动模式

### 在两个活动之间交替跳转

```java
Intent intent=new Intent(MainActivity2.this,MainActivity.class);
intent.setFlags(Intent.FLAG_ACTIVITY_CLEAR_TOP); //设置启动标志
startActivity(intent);
```

### 登录成功不在返回登录页面

```java
Intent intent=new Intent(this,LoginSuccessActivity.class);
 intent.setFlags(Intent.FLAG_ACTIVITY_CLEAR_TASK|Intent.FLAG_ACTIVITY_NEW_TASK);
startActivity(intent);
```

## Intent

显示intent就是上面使用的过程

隐式intent

```java
Intent intent=new Intent(this,LoginSuccessActivity.class);
intent.setAction(Intent.ACTION_DIAL);//跳转到拨号界面
Uri uri=Uri.parse("tel:"+"123");
intent.setData(uri);
startActivity(intent);
```

## Activity间传递数据

### 向下一个activity传递数据

```java
Intent intent=new Intent(this,LoginSuccessActivity.class);
Bundle bundle=new Bundle();
bundle.putString("key","value");
intent.putExtras(bundle);
startActivity(intent);
```

```java
Bundle bundle=getIntent().getExtras();
String value=bundle.getString("key");
```

### 向上一个activity传递数据

## 利用资源文件配置字符串

```java
String value=getString(R.string.app_name);
```

## CheckBox

android:button=“@null” //不要复选框的框

```xml
<CheckBox
        android:text="跑步"
        android:padding="5dp"
        android:button="@drawable/checkbox_selector"
        android:layout_width="wrap_content"
        android:layout_height="wrap_content">
    </CheckBox>
```

```xml
<?xml version="1.0" encoding="utf-8"?>
<selector xmlns:android="http://schemas.android.com/apk/res/android">
    <item android:state_checked="true" android:drawable="@drawable/ic_baseline_person_24"></item>
    <item  android:drawable="@drawable/ic_baseline_accessible_24"></item>
</selector>
```

```java
public void onCheckedChanged(CompoundButton compoundButton, boolean b) {
        String desc=String.format("您%s了这个复选框",b?"勾选":"取消");
        compoundButton.setText(desc);
    }
```

## RadioButton

```xml
<RadioGroup
            android:layout_width="wrap_content"
            android:layout_height="wrap_content"
            android:orientation="horizontal">
            <RadioButton
                android:text="男"
                android:checked="true"
                android:layout_width="wrap_content"
                android:layout_height="wrap_content">
            </RadioButton>
            <RadioButton
                android:text="女"
                android:layout_width="wrap_content"
                android:layout_height="wrap_content">
            </RadioButton>
        </RadioGroup>
```

## 文本监听器

```java
editText.addTextChangedListener(new TextWatcher() {
            public void beforeTextChanged(CharSequence charSequence, int i, int i1, int i2) {

            }
            public void onTextChanged(CharSequence charSequence, int i, int i1, int i2) {

            }
            public void afterTextChanged(Editable editable) {
                //文本框输入后改变
                String str=editText.getText().toString();
                if(str.length()==6)
                {
                    ViewUtil.hideOneInputMethod(MainActivity2.this,editText);
                }
            }
        });
```

```java
public class ViewUtil {
    public static void hideOneInputMethod(Activity activity, View view){
        //从系统服务中获取输入法管理器
        InputMethodManager imm=(InputMethodManager) activity.getSystemService(Context.INPUT_METHOD_SERVICE);
        //关闭屏幕上面的输入法软键盘
        imm.hideSoftInputFromWindow(view.getWindowToken(),0);
    }
}
```

## DatePicker

```
<DatePicker
        android:id="@+id/tv_date"
        android:datePickerMode="spinner"
        android:calendarViewShown="false"
        android:layout_width="match_parent"
        android:layout_height="wrap_content">
    </DatePicker>
```

```java
public void onClick(View view) {
        switch (view.getId())
        {
            case R.id.btn_ok:
                String desc=String.format("您选择的日期是%d年%d月%d日",tv_date.getYear(),tv_date.getMonth()+1,tv_date.getDayOfMonth());
                Log.e("hao",desc);
                tx_date.setText(desc);
                break;
            case R.id.btn_choose:
                //初始化为当前时间
                Calendar calendar= Calendar.getInstance();
                Log.e("hao",calendar.MONTH+"");
                DatePickerDialog dialog=new DatePickerDialog(this,this, calendar.get(Calendar.YEAR),calendar.get(Calendar.MONTH),calendar.get(Calendar.DAY_OF_MONTH));
                dialog.show();//显示日期对话框
                break;
        }
    }
    public void onDateSet(DatePicker datePicker, int year, int month, int datofMonth) {
        String desc=String.format("您选择的日期是%d年%d月%d日",year,month,datofMonth);
        Log.e("hao",desc);
        tx_date.setText(desc);
    }
```

## TimePicker

> 还可以在java代码中写时间选择器对话框

```xml
<TimePicker
        android:id="@+id/tv_time"
        android:layout_width="wrap_content"
        android:layout_height="wrap_content">
    </TimePicker>

```

## SharePreference

存储

```java
SharedPreferences preferences = getSharedPreferences("config", Context.MODE_PRIVATE);
SharedPreferences.Editor edit = preferences.edit();
        if(b)  //记住密码
        {
            edit.putString("phone",et_phone.getText().toString());
            edit.putString("password",et_password.getText().toString());
            edit.putBoolean("isChecked",b);
            edit.commit();  //提交
        }
        else
        {
            edit.putString("phone","");
            edit.putString("password","");
            edit.putBoolean("isChecked",b);
            edit.commit();  //提交
        }
```

读取

```java
String phone=preferences.getString("phone","");
String password=preferences.getString("password","");
Boolean isChecked=preferences.getBoolean("isChecked",false);
if(isChecked)
{
    et_phone.setText(phone);
    et_password.setText(password);
    ck_rememberr.setChecked(isChecked);
}
```

## SQLiteDatabase

```java
String databaseName=getFilesDir()+"/test.db";

public void onClick(View view) {
        if(view.getId()==R.id.btn_create_database)
        {
            SQLiteDatabase db=openOrCreateDatabase(databaseName, Context.MODE_PRIVATE,null);
            String desc=String.format("数据库%s创建%s",db.getPath(),(db!=null)?"成功":"失败");
            Toast.makeText(this,desc,Toast.LENGTH_SHORT).show();
            return;
        }
        else if(view.getId()==R.id.btn_delete_database)
        {
            boolean result=deleteDatabase(databaseName);
            String desc=String.format("数据库%s创建%s",databaseName, result?"成功":"失败");
            Toast.makeText(this,desc,Toast.LENGTH_SHORT).show();
            return;
        }
    }
```

## 外部存储空间

### 私有空间

```java
public void onClick(View view) {
        if(view.getId()==R.id.btn_save)
        {
            String name=et_name.getText().toString();
            String age=et_age.getText().toString();
            String height=et_height.getText().toString();
            String weight=et_weight.getText().toString();
            StringBuilder sb=new StringBuilder();
            sb.append("姓名").append(name).append('\n');
            sb.append("年龄").append(age).append('\n');
            sb.append("身高").append(height).append('\n');
            sb.append("体重").append(weight).append('\n');
            sb.append("婚否").append(ck_married.isChecked()?"是":"否").append('\n');
            String directory=null;
            //外部存储的私有空间
            String fileName=System.currentTimeMillis()+".txt";
            directory=getExternalFilesDir(Environment.DIRECTORY_DOWNLOADS).toString();
            path=directory+ File.separatorChar+fileName;
            FileUtil.saveText(path,sb.toString());
            Toast.makeText(this,"保存成功",Toast.LENGTH_SHORT).show();
            return;
        }
        else if(view.getId()==R.id.btn_read)
        {
            tv_txt.setText(FileUtil.openText(path));
        }
    }
```

### 公共空间

```java
//得开启权限
directory=Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DOWNLOADS).toString();
```

### 内部存储空间

```java
directory=getFilesDir().toString();
```

## Application内存存储

频繁使用的数据如手机号用户名等放入内存中

```java
private static MyApplication app;
public static MyApplication getInstance(){
        return app;
    }
MyApplication app=MyApplication.getInstance(); //获取实例
```

## 下拉列表

```xml
<Spinner
        android:id="@+id/sp_dropdown"
        android:layout_width="match_parent"
        android:spinnerMode="dropdown"
        android:layout_height="wrap_content">
    </Spinner>
```

```xml
<Spinner
        android:id="@+id/sp_dialog"
        android:layout_width="match_parent"
        android:spinnerMode="dialog"
        android:layout_height="wrap_content">
    </Spinner>
```

### simpleAdapter

```java
protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_spinner_icon);

        List<Map<String,Object>> list=new ArrayList<>();
        for(int i=0;i<iconArray.length;i++)
        {
            Map<String,Object> item=new HashMap<>();
            item.put("icon",iconArray[i]);
            item.put("name",starArray[i]);
            list.add(item);
        }
        SimpleAdapter simpleAdapter=new SimpleAdapter(this,list,R.layout.item_simple,
                new String[]{"icon","name"},new int[]{R.id.iv_icon,R.id.tv_show});
        Spinner sp_icon=findViewById(R.id.sp_icon);
        sp_icon.setAdapter(simpleAdapter);
        sp_icon.setSelection(0);
        sp_icon.setOnItemSelectedListener(this);
    }
```

### ListView

```xml
<ListView
        android:id="@+id/lv_show"
        android:divider="@null"
        android:dividerHeight="0dp"
        android:layout_width="match_parent"
        android:layout_height="wrap_content">
    </ListView>
```

### GridView

```xml
<GridView
        android:id="@+id/gv_show"
        android:numColumns="2"
        android:layout_width="match_parent"
        android:layout_height="wrap_content">
</GridView>
```

## ViewPager

```xml
<androidx.viewpager.widget.ViewPager
        android:id="@+id/vp_show"
        android:layout_width="match_parent"
        android:layout_height="370dp">
    </androidx.viewpager.widget.ViewPager>
```

```java
vp_show=findViewById(R.id.vp_show);
ImagePagerrAdapter adapter=new ImagePagerrAdapter(this,listView);
vp_show.setAdapter(adapter);
vp_show.addOnPageChangeListener(this);
```

## 广播

```java
public class MainActivity extends AppCompatActivity implements View.OnClickListener {
    private StandardReceiver standardReceiver;
    public static final String ORDER_ACTION="com.example.board.order";
    private  OrderAReceiver orderAReceiver;
    private OrderBReceiver orderBReceiver;
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        findViewById(R.id.btn_send).setOnClickListener(this);
        findViewById(R.id.byn_order).setOnClickListener(this);
    }

    public void onClick(View view) {
        if(view.getId()==R.id.btn_send)
        {
            //发送标准广播
            Intent intent=new Intent(StandardReceiver.STANDARD_ACTION);
            sendBroadcast(intent);
        }
        else if(view.getId()==R.id.byn_order)
        {
            Intent intent=new Intent(ORDER_ACTION);
            //发送有序广播 需要权限
            sendOrderedBroadcast(intent,null);
        }
    }

    public void onStart() {
        super.onStart();
        standardReceiver=new StandardReceiver();
        //创建一个意图过滤器 只处理STANDARD_ACTION的广播
        IntentFilter filter=new IntentFilter(StandardReceiver.STANDARD_ACTION);
        registerReceiver(standardReceiver,filter);

        //多个接收器处理有序广播顺序 优先级越大 先收到 优先级相同 越早注册 先接收
        orderAReceiver=new OrderAReceiver();
        IntentFilter filterA=new IntentFilter(ORDER_ACTION);
        filterA.setPriority(8);
        registerReceiver(orderAReceiver,filterA);

        orderBReceiver=new OrderBReceiver();
        IntentFilter filterB=new IntentFilter(ORDER_ACTION);
        filterB.setPriority(10);
        registerReceiver(orderBReceiver,filterB);
    }
    public void onStop() {
        super.onStop();
        //注销 取消接收
        unregisterReceiver(standardReceiver);

        //取消有序
        unregisterReceiver(orderAReceiver);
        unregisterReceiver(orderBReceiver);
    }
}
```

```java
public class StandardReceiver extends BroadcastReceiver {
    public static final String STANDARD_ACTION="com.example.board.standard";
    //一旦收到标准广播 马上触发
    public void onReceive(Context context, Intent intent) {
        if(intent!=null &&intent.getAction().equals(STANDARD_ACTION))
        {
            Log.e("hao","收到一个广播");
        }
    }
}
```

```java
public class OrderBReceiver extends BroadcastReceiver {
    @Override
    public void onReceive(Context context, Intent intent) {
        if(intent!=null&&intent.getAction().equals(MainActivity.ORDER_ACTION))
        {
            Log.e("hao","接收器B收到一个有序广播");
            abortBroadcast();  //中断广播 后面的无法接收
        }
    }
}
```



# 前端框架XUI

[饭后Android 第三餐-XUI框架（XUI介绍，使用方法，控件使用（九个Button，导航栏，可伸缩布局，顶部弹出框））_android xui-CSDN博客](https://blog.csdn.net/qq_46526828/article/details/108904260)

[Button · xuexiangjys/XUI Wiki (github.com)](https://github.com/xuexiangjys/XUI/wiki/Button)

## Button

```xml
<com.xuexiang.xui.widget.button.ButtonView
       style="@style/ButtonView.Green"
        android:id="@+id/task2_btn"
        android:layout_width="wrap_content"
        android:layout_height="wrap_content"
        android:textSize="30sp"
        app:textRadius="10dp"
        />
```



```xml
<com.xuexiang.xui.widget.button.roundbutton.RoundButton
            style="@style/RoundButton.Auto"
    		android:id="@+id/task2_btn"
            android:layout_width="wrap_content"
            android:layout_height="wrap_content"
            android:textSize="30sp"
            app:textRadius="10dp"
            android:text="默认圆角大小"
            android:enabled="false"/>
```



```xml
<com.xuexiang.xui.widget.button.shinebutton.ShineButton
            android:id="@+id/shine_button"
            android:layout_width="30dp"
            android:layout_height="30dp"
            android:layout_gravity="center"
            app:sb_checked_color="#f26d7d"
            app:sb_icon_image="@drawable/ic_pre"
            app:sb_normal_color="@android:color/darker_gray" />
```



## Spinner

```xml
 <com.xuexiang.xui.widget.spinner.materialspinner.MaterialSpinner
            style="@style/Material.SpinnerStyle"
            android:layout_marginLeft="20dp"
            android:layout_width="wrap_content"
            android:layout_height="wrap_content"
            android:textSize="20sp"
            android:enabled="true"
            app:ms_entries="@array/mood_array"
            />
```

